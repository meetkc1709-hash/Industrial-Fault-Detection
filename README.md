# Industrial Fault Detection

Big Data Analytics Project Submission 5: a Streamlit dashboard and machine fault
prediction application using the AI4I 2020 dataset and an existing Random Forest.

## Run locally

Use Python 3.12 and a virtual environment:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app resolves its dataset and saved model paths relative to `app.py`, so it
does not depend on the shell working directory. No secrets are required.

## Saved model

Keep `models/random_forest_model.pkl` and `models/feature_columns.pkl` together.
The existing model loads successfully with scikit-learn 1.9.1; keep that version
pinned because loading scikit-learn models across versions is unsupported.
Load only trusted pickle/joblib files. Deployment does not train the model.

`train_model.py` retrains and overwrites both saved model files when executed.
Do not run it to launch or deploy the website. For optional local data exploration,
install `requirements-dev.txt`. Run the training and exploration scripts from the
project root; these scripts use working-directory-relative paths.

## Verify

```powershell
python -m pip check
python -m unittest discover -s tests -v
```

The smoke tests run the app from a temporary working directory and exercise
predictions for L, M, and H products using the saved model.

## Deploy to Streamlit Community Cloud

1. Push this project to the GitHub repository `Industrial-Fault-Detection`.
2. Sign in at https://share.streamlit.io/ with GitHub and grant repository access.
3. Choose **Create app**, select the repository, branch `main`, and file `app.py`.
4. In **Advanced settings**, select Python **3.12** to match the verified environment.
5. Choose an available app subdomain and deploy. Community Cloud installs the
   root `requirements.txt`; no training command or system packages are required.
6. Wait for the app to start, open its public URL, and test the prediction button.
   Record the actual assigned `https://<subdomain>.streamlit.app` URL for submission.

Do not commit `.venv`, `.idea`, environment files, or `.streamlit/secrets.toml`.
If secrets are needed later, enter them through Community Cloud's app settings.

Official deployment guide:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Interpretation

The AI4I dataset is synthetic. The displayed failure score is a model estimate,
not a validated real-world equipment failure probability.
