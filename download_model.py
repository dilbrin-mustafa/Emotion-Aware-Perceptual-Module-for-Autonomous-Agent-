import os
import urllib.request
import ssl

def download_clean():
    # Official Hugging Face ONNX Model Zoo mirror
    url = "https://huggingface.co/onnxmodelzoo/emotion-ferplus-8/resolve/main/emotion-ferplus-8.onnx"
    filename = "emotion_model.onnx"
    
    # Context to bypass some strict SSL certification errors on corporate/school networks
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    if os.path.exists(filename):
        print(f"Removing old/corrupted {filename}...")
        os.remove(filename)
        
    print(f"Downloading from Hugging Face: {url}...")
    try:
        # Use a User-Agent to avoid 403 Forbidden errors
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, filename)
        
        size = os.path.getsize(filename)
        print(f"Download complete. Size: {size/1024/1024:.2f} MB")
        
        if size < 1000:
             print("WARNING: File is too small. It might be a text file (LFS pointer).")
        else:
             print("SUCCESS: Model is ready.")
             
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    download_clean()