from concurrent.futures import ThreadPoolExecutor
import pdfplumber
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from groq import Groq
import asyncio
import json
import os
import re
import shutil
import io 
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from youtube_transcript_api import YouTubeTranscriptApi as YTService
from ..database import get_db
from ..models import models
from ..schemas import schemas
import time
from huggingface_hub import batch_bucket_files,HfApi
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0" 
load_dotenv()
my_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")
HF_BUCKET_ID = os.getenv("HF_BUCKET_ID", "your-username/academic-vault")

app = FastAPI()
my_workers = ThreadPoolExecutor(max_workers=4)
ai_model = SentenceTransformer("all-MiniLM-L6-v2")

@app.post("/bulk-courses")
def add_many_courses(
    data: schemas.BulkCourseCreate, db: Session = Depends(get_db)
):
    all_courses = [models.Course(title=item.title) for item in data.courses]
    db.add_all(all_courses)
    db.commit()
    return {"message": f"Successfully added {len(all_courses)} courses"}

@app.post("/bulk-sections")
def add_many_sections(
    data: schemas.BulkSectionCreate, db: Session = Depends(get_db)
):
    all_sections = [
        models.Section(title=item.title, course_id=item.course_id)
        for item in data.sections
    ]
    db.add_all(all_sections)
    db.commit()
    return {"message": f"Successfully added {len(all_sections)} sections"}

@app.post("/bulk-lessons")
def add_many_lessons(
    data: schemas.BulkLessonCreate, db: Session = Depends(get_db)
):
    all_lessons = [
        models.Lesson(
            title=item.title, content=item.content, section_id=item.section_id
        )
        for item in data.lessons
    ]
    db.add_all(all_lessons)
    db.commit()
    return {"message": f"Successfully added {len(all_lessons)} lessons"}

@app.get("/get-all-data")
def show_everything(db: Session = Depends(get_db)):
    data = db.query(models.Course).all()
    return data

def get_vector_list(my_text: str) -> list:
    if not my_text or not my_text.strip():
        return [0.0] * 384
    return ai_model.encode(my_text).tolist()

def break_pdf_into_chunks(
    pdf_path: str, max_size: int = 5000, overlap_size: int = 1200
) -> list:
    full_text = ""
    syllabus_signals = [
        "syllabus", "curriculum", "course contents", "topics", "unit", "chapter", 
        "module", "examination", "study material", "academic", "course outline", 
        "subject matter", "course structure",
    ]
    noise_keywords = [
        "grading policy", "office hours", "attendance rules", "how to apply", 
        "eligibility criteria", "rulebook", "terms and conditions", "candidate instructions",
    ]
    with pdfplumber.open(pdf_path) as pdf_file:
        for single_page in pdf_file.pages:
            page_text = single_page.extract_text(layout=False)
            if page_text:
                lower_text = page_text.lower()
                is_syllabus_page = any(word in lower_text for word in syllabus_signals)
                if is_syllabus_page:
                    clean_lines = []
                    for line in page_text.splitlines():
                        if not any(word in line.lower() for word in noise_keywords):
                            clean_lines.append(line)
                    full_text += "\n".join(clean_lines) + "\n"
                
    full_text = re.sub(r"[ \t]+", " ", full_text)
    if len(full_text.strip()) < 100:
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf_file:
            for single_page in pdf_file.pages:
                page_text = single_page.extract_text(layout=False)
                if page_text:
                    full_text += page_text + "\n"
        full_text = re.sub(r"[ \t]+", " ", full_text)
    final_chunks = []
    current_index = 0
    total_length = len(full_text)
    if total_length <= max_size:
        return [full_text.strip()]
    while current_index < total_length:
        end_index = min(current_index + max_size, total_length)
        small_chunk = full_text[current_index:end_index]
        final_chunks.append(small_chunk.strip())
        current_index += max_size - overlap_size
    return final_chunks

def get_course_title(first_chunk: str) -> str:
    prompt = f"Analyze the text and extract ONLY the main course or exam title name. Return nothing but the raw string.\nText: {first_chunk[:1000]}"
    with Groq(api_key=my_key) as client:
        try:
            api_result = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.0,
            )
            return api_result.choices[0].message.content.strip().replace('"', "")
        except Exception as error_msg:
            if "rate_limit_exceeded" in str(error_msg).lower():
                time.sleep(2.0)
                try:
                    api_result = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama-3.1-8b-instant",
                        temperature=0.0,
                    )
                    return api_result.choices[0].message.content.strip().replace('"', "")
                except Exception:
                    return "Comprehensive Academic Syllabus"
            return "Comprehensive Academic Syllabus"

def get_sections_and_lessons(chunk_text: str):
    prompt = f"""
    Analyze the following text excerpt and identify if it contains an academic curriculum, exam topics, or educational modules.
    CRITICAL FILTRATION & EXTRACTION RULES:
    1. If this text contains ONLY general information, background history, guidelines, cover pages, application forms, instructions to candidates, eligibility criteria, or non-educational content, you MUST return an empty sections list: {{"sections": []}}.
    2. Only extract data if there are explicit, real academic subjects, units, chapters, or exam study topics.
    3. DO NOT summarize, paraphrase, or invent descriptions for the topics.
    4. Under each valid section, extract the exact topic names, keywords, or subject terms exactly as they appear in the PDF and place them as strings inside the 'lessons' array.
    5. Extract every single topic. Do not leave out or skip any academic material.
    Your response must match this JSON schema format perfectly:
    {{
        "sections": [
            {{
                "title": "Name of the unit, chapter, or core module theme",
                "lessons": [
                    "Exact Name of Topic 1 from PDF",
                    "Exact Name of Topic 2 from PDF"
                ]
            }}
        ]
    }}
    TEXT EXCERPT FOR TESTING AND EXTRACTION:
    {chunk_text}
    """
    with Groq(api_key=my_key) as client:
        try:
            api_result = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an exhaustive syllabus parsing machine. Your priority is to extract only real academic curriculum topics and discard unrelated background text. You must return your response as a valid JSON object.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model="llama-3.1-8b-instant",
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return json.loads(api_result.choices[0].message.content)
        except Exception as error_msg:
            if "rate_limit_exceeded" in str(error_msg).lower():
                time.sleep(2.5)
                api_result = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an exhaustive syllabus parsing machine. Your priority is to extract only real academic curriculum topics and discard unrelated background text. You must return your response as a valid JSON object.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model="llama-3.1-8b-instant",
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                return json.loads(api_result.choices[0].message.content)
            raise error_msg

def upload_file_to_hf_bucket(local_file_path: str, remote_path: str):
    """
    Safely uploads files to Hugging Face using the robust, official HfApi class.
    Bypasses structural Xet tree authentication bugs.
    """
    try:
        print(f"[HF Hub] Preparing secure upload to {HF_BUCKET_ID}/{remote_path}...")
        api = HfApi(token=hf_token)
        
        api.upload_file(
            path_or_fileobj=local_file_path,
            path_in_repo=remote_path,
            repo_id=HF_BUCKET_ID,
            repo_type="model"  
        )
        print(f"[HF Hub] Successfully pushed asset: {remote_path}")
    except Exception as e:
        print(f"[HF Hub Error] Direct upload layer failure: {e}")
        raise e


@app.post("/process-syllabus")
async def start_syllabus_processing(
    file: UploadFile = File(...), db: Session = Depends(get_db)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Please upload a valid PDF document."
        )
    saved_file_name = f"temp_{file.filename}"
    with open(saved_file_name, "wb") as local_disk_file:
        shutil.copyfileobj(file.file, local_disk_file)
    try:
        my_loop = asyncio.get_event_loop()
        all_text_chunks = await my_loop.run_in_executor(
            my_workers, break_pdf_into_chunks, saved_file_name
        )
        if not all_text_chunks:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text content from the PDF.",
            )
        course_name = get_course_title(all_text_chunks[0])
        course_vector_values = get_vector_list(course_name)
        new_course_row = models.Course(
            title=course_name, course_embedding=course_vector_values
        )
        db.add(new_course_row)
        db.flush()
        output_data = {"course_title": course_name, "sections": []}
        seen_sections = set()
        seen_lessons = set()
        chunk_counter = 0
        for current_chunk in all_text_chunks:
            chunk_counter += 1
            if chunk_counter > 1 and chunk_counter % 2 == 0:
                await asyncio.sleep(1.5)
            json_response = await my_loop.run_in_executor(
                my_workers, get_sections_and_lessons, current_chunk
            )
            for item in json_response.get("sections", []):
                section_name = item.get("title", "Untitled Section").strip()
                section_unique_key = section_name.lower()

                if section_unique_key not in seen_sections:
                    sec_vector_values = get_vector_list(section_name)
                    new_section_row = models.Section(
                        title=section_name,
                        course_id=new_course_row.id,
                        section_embedding=sec_vector_values,
                    )
                    db.add(new_section_row)
                    db.flush()
                    seen_sections.add(section_unique_key)
                    new_section_dict = {"title": section_name, "lessons": []}
                    output_data["sections"].append(new_section_dict)
                else:
                    new_section_dict = next(
                        s
                        for s in output_data["sections"]
                        if s["title"].lower() == section_unique_key
                    )
                    new_section_row = (
                        db.query(models.Section)
                        .filter(
                            models.Section.course_id == new_course_row.id,
                            models.Section.title.ilike(section_name),
                        )
                        .first()
                    )
                    if not new_section_row:
                        sec_vector_values = get_vector_list(section_name)
                        new_section_row = models.Section(
                            title=section_name,
                            course_id=new_course_row.id,
                            section_embedding=sec_vector_values,
                        )
                        db.add(new_section_row)
                        db.flush()
                for single_lesson in item.get("lessons", []):
                    if isinstance(single_lesson, dict):
                        lesson_name = single_lesson.get(
                            "title", "Untitled Topic"
                        ).strip()
                    else:
                        lesson_name = str(single_lesson).strip()
                    if not lesson_name:
                        continue
                    lesson_unique_key = (
                        f"{new_section_row.id}::{lesson_name.lower()}"
                    )
                    if lesson_unique_key not in seen_lessons:
                        les_vector_values = get_vector_list(lesson_name)
                        new_lesson_row = models.Lesson(
                            title=lesson_name,
                            content=lesson_name,
                            section_id=new_section_row.id,
                            lesson_embedding=les_vector_values,
                        )
                        db.add(new_lesson_row)
                        db.flush()
                        seen_lessons.add(lesson_unique_key)
                        new_section_dict["lessons"].append(lesson_name)
        db.commit()

        safe_course_title = re.sub(r'[^a-zA-Z0-9_]', '_', course_name)
        remote_syllabus_path = f"syllabus/{safe_course_title}_{new_course_row.id}.pdf"
        
        await my_loop.run_in_executor(
            None, upload_file_to_hf_bucket, saved_file_name, remote_syllabus_path
        )
        return {"status": "Success", "data": output_data}
    except Exception as error_msg:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Syllabus Processing Matrix Error: {str(error_msg)}",
        )
    finally:
        if os.path.exists(saved_file_name):
            os.remove(saved_file_name)
def parse_youtube_id(video_url: str) -> str:
    video_url = video_url.strip()
    regex_pattern = r"(?:v=|\/shorts\/|\/embed\/|\/live\/|\/v\/|youtu\.be\/|\/v=|^)([a-zA-Z0-9_-]{11})"
    regex_match = re.search(regex_pattern, video_url)
    if not regex_match:
        raise ValueError(
            "The provided link does not match a valid YouTube URL format."
        )
    return regex_match.pattern, regex_match.group(1)[1] if isinstance(regex_match.group(1), tuple) else regex_match.group(1)
def generate_notes_from_text(transcript_content: str) -> str:
    block_size = 4500
    my_chunks = []
    for i in range(0, len(transcript_content), block_size):
        my_chunks.append(transcript_content[i : i + block_size])
    if len(my_chunks) > 4:
        my_chunks = my_chunks[:4]
    saved_notes_blocks = []
    
    with Groq(api_key=my_key) as client:
        for index_num, data_chunk in enumerate(my_chunks):
            prompt = f"""
You are an expert academic research assistant. Analyze this section ({index_num + 1}/{len(my_chunks)}) of an educational lecture transcript.
Provide highly detailed study notes in clean Markdown layout. 
CRITICAL LANGUAGE RULE:
- Your output MUST be written entirely in English. 
- If the input transcript is in Hindi or Hinglish, translate the core technical concepts and explanations cleanly into professional English.
Focus strictly on definitions, core concepts, algorithms, or code paradigms mentioned.
Do not add introductory fluff like "In this segment...".
TRANSCRIPT FRAGMENT:
{data_chunk}
"""
            try:
                api_result = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You output dense, high-quality technical study notes in markdown layout.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model="llama-3.1-8b-instant",
                    temperature=0.2,
                )
                saved_notes_blocks.append(
                    api_result.choices[0].message.content.strip()
                )
            except Exception as error_msg:
                if "rate_limit_exceeded" in str(error_msg).lower():
                    time.sleep(2.0)
                    api_result = client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "You output dense technical notes.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        model="llama-3.1-8b-instant",
                        temperature=0.2,
                    )
                    saved_notes_blocks.append(
                        api_result.choices[0].message.content.strip()
                    )
                else:
                    raise error_msg
    final_merged_string = "\n\n---\n\n".join(saved_notes_blocks)
    return final_merged_string

@app.post("/process-youtube")
async def start_youtube_processing(
    payload: schemas.YouTubeProcessRequest, db: Session = Depends(get_db)
):
    try:
        regex_pattern = r"(?:v=|\/shorts\/|\/embed\/|\/live\/|\/v\/|youtu\.be\/|\/v=|^)([a-zA-Z0-9_-]{11})"
        regex_match = re.search(regex_pattern, payload.url.strip())
        if not regex_match:
            raise HTTPException(status_code=400, detail="The provided link does not match a valid YouTube URL format.")
        yt_id = regex_match.group(1)
        
        def run_api_fetch():
            yt_service_tool = YTService()
            try:
                available_scripts = yt_service_tool.list(yt_id)
                picked_script = available_scripts.find_transcript(["en", "hi"])
                raw_lines = picked_script.fetch()
                joined_string = " ".join(
                    [item.text for item in raw_lines]
                )
                return re.sub(r"\s+", " ", joined_string).strip()
            except Exception as inner_error:
                try:
                    raw_lines = yt_service_tool.fetch(yt_id)
                    joined_string = " ".join(
                        [item.text for item in raw_lines]
                    )
                    return re.sub(r"\s+", " ", joined_string).strip()
                except Exception:
                    raise inner_error
        my_loop = asyncio.get_event_loop()
        full_transcript_text = await my_loop.run_in_executor(
            my_workers, run_api_fetch
        )
        if not full_transcript_text:
            raise HTTPException(
                status_code=400,
                detail="The extracted transcript text content is empty.",
            )
        async def save_into_database():
            all_markdown_notes = generate_notes_from_text(full_transcript_text)
            notes_vector_values = get_vector_list(all_markdown_notes)
            if hasattr(models, 'VideoSummary'):
                new_summary_row = models.VideoSummary(
                    video_id=yt_id,
                    video_url=payload.url,
                    raw_transcript=full_transcript_text,
                    detailed_notes=all_markdown_notes,
                    summary_embedding=notes_vector_values,
                )
                db.add(new_summary_row)
            else:
                fallback_section_id = getattr(payload, 'section_id', None)
                if not fallback_section_id:
                    any_section = db.query(models.Section).first()
                    fallback_section_id = any_section.id if any_section else 1
                new_summary_row = models.Lesson(
                    title=f"AI Video Notes ({yt_id})",
                    content=all_markdown_notes,
                    section_id=fallback_section_id,
                    lesson_embedding=notes_vector_values
                )
                db.add(new_summary_row)
            db.flush()
            db.commit()
            return all_markdown_notes
        summary_output = await asyncio.shield(save_into_database())
        local_notes_file = f"notes_{yt_id}.md"
        with open(local_notes_file, "w", encoding="utf-8") as nf:
            nf.write(summary_output)
            
        await my_loop.run_in_executor(
            None, upload_file_to_hf_bucket, local_notes_file, f"youtube_notes/YT_{yt_id}.md"
        )
    
        if os.path.exists(local_notes_file):
            os.remove(local_notes_file)
        return {
            "summary": summary_output
        }
    except HTTPException as status_err:
        raise status_err
    except ValueError as input_error:
        raise HTTPException(status_code=400, detail=str(input_error))
    except Exception as general_error:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"YouTube standalone pipeline processing breakdown: {str(general_error)}",
        )