from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import traceback
import psycopg
from routes import get_routers

app = FastAPI(redirect_slashes=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware('http')
async def custom_response(request: Request, call_next):
    if request.method == 'OPTIONS':
        return Response(
            None,
            status_code=200,
            headers={
                'Allow': 'GET, POST, OPTIONS, HEAD, DELETE, PUT',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, HEAD, DELETE, PUT',
                'Access-Control-Max-Age': '86400',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': '*',
            }
        )
    
    start_time = time.time()
    
    try:
        resp: Response = await call_next(request)

    except psycopg.OperationalError:
        traceback.print_exc()
        resp = JSONResponse(status_code=500, content={"detail": "Database communication link failure."})

    except psycopg.errors.UniqueViolation:
        traceback.print_exc()
        resp = JSONResponse(status_code=403, content={"detail": "A record already exists with that information."})

    except psycopg.errors.IntegrityConstraintViolation:
        traceback.print_exc()
        resp = JSONResponse(status_code=403, content={"detail": "Failed to perform action: Integrity violation."})

    except psycopg.DatabaseError:
        traceback.print_exc()
        resp = JSONResponse(status_code=500, content={"detail": "Unspecified database-related error."})

    process_time = time.time() - start_time
    resp.headers['X-Process-Time'] = f"{int(process_time * 1000)}ms"
    
    return resp

for router in get_routers():
    app.include_router(router)
