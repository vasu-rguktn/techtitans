from gan_auth_inference import detect_image

# Change this path to test image
# image_path = r"D:\Desktop\gan_fake_dataset\gan_fake_27.png"-True
# image_path=r"C:\Users\pooji\Desktop\majoprojec t\fake_xrays\11_IM-0067-1001.dcm.png"-False
# image_path=r"C:\Users\pooji\Desktop\majoprojec t\fake_autoencoder1\13_IM-0198-1001.dcm.png"-True
# image_path=r"C:\Users\pooji\Desktop\majoprojec t\unseen_fake_test\2402_IM-0951-2001.dcm.png"-False
# image_path=r"D:\Fake_images\elastic\51_IM-2125-1001.dcm.png"-False
# image_path=r"D:\Fake_images\blur\110_IM-0067-1001.dcm.png" #real
# image_path=r"C:\Users\pooji\Desktop\majoprojec t\archive\images\images_normalized\8_IM-2333-1001.dcm.png"
# image_path=r"C:\Users\pooji\Desktop\majoprojec t\fake_xrays\14_IM-0256-1001.dcm.png" #real
label, prob = detect_image(image_path)

print("\n=== GAN Authenticity Check ===")
print(f"Prediction: {label}")
print(f"Fake Probability: {prob:.4f}")