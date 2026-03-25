"""Chat API route - Handles chat/completion requests."""

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.schemas.request import ChatRequest, CompletionRequest
from src.api.schemas.response import ChatResponse, CompletionResponse, StreamChunk
from src.api.dependencies.auth import get_current_user
from src.services.chat_service import ChatService
from src.services.streaming_service import StreamingService

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequestPayload(BaseModel):
    """Chat request payload."""

    messages: Optional[List[dict]] = Field(
        None, description="List of message objects with 'role' and 'content'"
    )
    prompt: Optional[str] = Field(
        None, description="Input prompt (alternative to messages)"
    )
    model: Optional[str] = Field(None, description="Model name to use")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(
        None, ge=1, le=8192, description="Maximum tokens to generate"
    )
    stream: bool = Field(False, description="Enable streaming response")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")


class CompletionRequestPayload(BaseModel):
    """Completion request payload."""

    prompt: str = Field(..., description="Input prompt")
    model: Optional[str] = Field(None, description="Model name to use")
    suffix: Optional[str] = Field(None, description="Suffix to append after completion")
    max_tokens: int = Field(512, ge=1, le=8192, description="Max tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: int = Field(50, ge=0, description="Top-k sampling parameter")
    repetition_penalty: float = Field(
        1.1, ge=1.0, le=2.0, description="Repetition penalty"
    )
    stream: bool = Field(False, description="Enable streaming response")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")


@router.post("/completions", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequestPayload,
    background_tasks: BackgroundTasks,
    
):
    """Create a text completion."""
    try:
        chat_service = ChatService()
        
        if request.stream:
            return StreamingResponse(
                chat_service.stream_completion(
                    prompt=request.prompt,
                    model=request.model,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    stop=request.stop,
                ),
                media_type="text/event-stream",
            )
        
        result = await chat_service.create_completion(
            prompt=request.prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            stop=request.stop,
        )
        
        return CompletionResponse(
            id=f"cmpl-{result['id']}",
            object="text_completion",
            created=result["created"],
            model=result["model"],
            choices=[
                {
                    "text": result["text"],
                    "index": 0,
                    "finish_reason": result["finish_reason"],
                }
            ],
            usage={
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/completions")
async def chat_completions(request: ChatRequestPayload):
    """OpenAI-compatible chat completions endpoint."""
    try:
        chat_service = ChatService()
        
        # Convert prompt to messages format if provided
        messages = request.messages
        if request.prompt and not messages:
            messages = [{"role": "user", "content": request.prompt}]
        
        if not messages:
            raise HTTPException(status_code=400, detail="Either 'messages' or 'prompt' is required")
        
        if request.stream:
            return StreamingResponse(
                chat_service.stream_chat(
                    messages=messages,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    stop=request.stop,
                ),
                media_type="text/event-stream",
            )
        
        result = await chat_service.chat(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop=request.stop,
        )
        
        return ChatResponse(
            id=f"chatcmpl-{result['id']}",
            object="chat.completion",
            created=result["created"],
            model=result["model"],
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["content"],
                    },
                    "finish_reason": result["finish_reason"],
                }
            ],
            usage={
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_chat_history(
    limit: int = 50,
    
):
    """Get chat history for the current user."""
    # Placeholder - would integrate with storage
    return {"history": [], "limit": limit}


@router.delete("/history/{session_id}")
async def clear_chat_history(
    session_id: str,
    
):
    """Clear chat history for a specific session."""
    return {"status": "cleared", "session_id": session_id}