from sqlalchemy import Column, Integer, String, ForeignKey, Text,DateTime,func
from sqlalchemy.orm import relationship
from ..database import Base
from pgvector.sqlalchemy import Vector  

class Course(Base):
    __tablename__ = "courses"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    course_embedding = Column(Vector(384), nullable=True)
    sections = relationship("Section", back_populates="course", cascade="all, delete-orphan")

class Section(Base):
    __tablename__ = "sections"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    section_embedding = Column(Vector(384), nullable=True)
    course = relationship("Course", back_populates="sections")
    lessons = relationship("Lesson", back_populates="section", cascade="all, delete-orphan")

class Lesson(Base):
    __tablename__ = "lessons"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=True) 
    section_id = Column(Integer, ForeignKey("sections.id"),nullable=False)
    section = relationship("Section", back_populates="lessons")
    lesson_embedding = Column(Vector(384), nullable=True)

class VideoSummary(Base):
    __tablename__ = "video_summaries"
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String(11), nullable=False, index=True)
    video_url = Column(String(255), nullable=False)
    raw_transcript = Column(Text, nullable=True)
    detailed_notes = Column(Text, nullable=False)
    summary_embedding = Column(Vector(384), nullable=True)