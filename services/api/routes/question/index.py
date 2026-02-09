from fastapi import APIRouter, HTTPException
from libs.services.question_service import process_question
from libs.structures.question import QuestionRequest, QuestionResponse

router = APIRouter(tags=["question"])

@router.post("/", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    try:
        answer, references = process_question(request.question)
        
        return QuestionResponse(
            answer=answer,
            references=references
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")