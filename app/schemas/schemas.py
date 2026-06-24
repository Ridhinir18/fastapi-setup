
from pydantic import BaseModel
from typing import List, Optional
class CourseCreate(BaseModel):
    title: str
class SectionCreate(BaseModel):
    title: str
    course_id: int
class LessonCreate(BaseModel):
    title: str
    content: Optional[str] = None
    file_path: Optional[str] = None
    section_id: int
class BulkCourseCreate(BaseModel):
    courses: List[CourseCreate]
class BulkSectionCreate(BaseModel):
    sections: List[SectionCreate]
class BulkLessonCreate(BaseModel):
    lessons: List[LessonCreate]
class LessonData(BaseModel):
    title: str
    content: str
class SectionData(BaseModel):
    title: str
    lessons: List[LessonData]
class SyllabusResponse(BaseModel):
    course_title: str
    sections: List[SectionData]
class YouTubeProcessRequest(BaseModel):
    url: str
