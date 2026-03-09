from deepface import DeepFace

result = DeepFace.verify(
    img1_path="image/img2.jpeg",
    img2_path="image/4.jpg",
    model_name="ArcFace",      # Best for aging & time gap
    detector_backend="retinaface"  # Most accurate detector
)

if result["verified"]:
    print("✅ MATCH (Same person)")
else:
    print("❌ NOT A MATCH (Different persons)")

print(result)
