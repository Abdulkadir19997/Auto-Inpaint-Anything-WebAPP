from fastapi import FastAPI, APIRouter
import uvicorn
from app.routes.dino_sam import router as dino_sam_router
from app.routes.remove_anything import router as remove_anything_router
from app.routes.replace_anything import router as replace_anything_router
from app.routes.fill_anything import router as fill_anything_router


app = FastAPI(title="Auto Inpainter API")


router = APIRouter(prefix="/app/demo")

# Middleware'ı ekleyin
# app.middleware("http")(catch_exceptions)
router.include_router(dino_sam_router, prefix="/dino_sam", tags=["Object detection and Segmentation"])
router.include_router(remove_anything_router, prefix="/remove_anything", tags=["LaMa inpainter"])
router.include_router(replace_anything_router, prefix="/replace_anything", tags=["SD Background converter"])
router.include_router(fill_anything_router, prefix="/fill_anything", tags=["SD object converter"])
# # dependencies=[Depends(JWTBearer())]  authentication for api add

app.include_router(router)
# Run the application using Uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5004, reload=True)
