# CMPT742-Visual Computing Lab I - Media Hand Tracking
## Dataset: https://facundoq.github.io/datasets/lsa64/

A serverless sign-language translation API deployed on Google Cloud Run. This service uses MediaPipe to process videos containing a single sign-language gesture, detects the hand-skeleton time series, and returns the corresponding textual translation.

You can see use the sign-language translation API here: https://hand-skeleton-api-725163779003.us-west1.run.app/

---

## Prerequisites

1. **Google Cloud SDK** (`gcloud`) installed locally  
2. **Docker** (for local builds, optional)  
3. A Google Cloud project (e.g. `cmpt756teamproject`) with billing enabled  
4. IAM roles:  
   - **Cloud Build Editor**  
   - **Cloud Run Admin**  
   - **Storage Admin** (if you use GCS for static hosting)  
5. (Optional) Python 3.7+ and a `requirements.txt` listing your dependencies, including `mediapipe`, `fastapi`, and `uvicorn`.

---

## Setup

1. **Initialize `gcloud` and authenticate**

   ```bash
   gcloud init
   ```

   * Log in with your Google account and select the `YourProjectName` project.

2. **Set the active project**

   ```bash
   gcloud config set project YourProjectName
   ```

3. **Enable required APIs**

   ```bash
   gcloud services enable \
     cloudbuild.googleapis.com \
     run.googleapis.com \
     containerregistry.googleapis.com
   ```

4. **Ensure billing is enabled** for your project. Cloud Build and Cloud Run require an active billing account.

5. **(Optional) Configure Docker credentials**

   ```bash
   gcloud auth configure-docker
   ```

   This allows local `docker push`/`docker pull` to Google Container Registry.

6. **Verify IAM permissions**

   * Your user or service account needs the following roles:

     * Cloud Build Editor
     * Cloud Run Admin
     * Storage Admin (for Container Registry)

7. **Prepare a `Dockerfile`** in the root of this repository. Example:

   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY . /app
   RUN pip install --no-cache-dir -r requirements.txt
   CMD ["python", "main.py"]
   ```
   
---

## Build & Deploy

1. **Build the Docker image and push to Container Registry**:

   ```bash
   gcloud builds submit --tag gcr.io/YourProjectName/hand-skeleton-api
   ```

2. **Deploy to Cloud Run**:

   ```bash
   gcloud run deploy hand-skeleton-api \
     --image gcr.io/YourProjectName/hand-skeleton-api \
     --platform managed \
     --region us-west1 \
     --allow-unauthenticated \
     --memory=2Gi \
     --cpu=2
   ```
   
---

## Usage

After deployment, you'll receive a service URL. You can use that URL to open the GUI of sign-language translator.
