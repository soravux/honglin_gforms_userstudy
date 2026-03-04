import glob
import os
import pickle
import re

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

SCOPES = [
    'https://www.googleapis.com/auth/forms.body',
    'https://www.googleapis.com/auth/drive.file',
]

# FORM_ID = '1h_uxEBPYfqamCLNBkALTvYDJzXUOCEzR3YDVyvfgy2M'
# IMAGE_DIR = '../data/v1/user_study_images'

FORM_ID = '1bqfU9-n-0eGz1W2pVrWConvo3pK8mNp6_yARyPczfPQ'
IMAGE_DIR = '../data/v2/user_study_images'

QUESTIONS = [
    'Which set of poses appears more anatomically plausible and natural for the given human or animal mesh?',
    'Which set of poses is more diverse, with a larger range of different poses? Ignore completely unrecognizable poses.',
]
CHOICES = [{'value': 'Method A'}, {'value': 'Method B'}]


def get_credentials():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as f:
            creds = pickle.load(f)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            # creds = service_account.Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
        with open('token.pickle', 'wb') as f:
            pickle.dump(creds, f)
    return creds


def discover_images(directory):
    """Find all question images and return them sorted numerically."""
    paths = glob.glob(os.path.join(directory, 'question_*.jpg'))
    def sort_key(p):
        m = re.search(r'question_(\d+)_', os.path.basename(p))
        return int(m.group(1)) if m else 0
    return sorted(paths, key=sort_key)


def upload_to_drive(drive_service, local_path):
    """Upload a local image to Google Drive and return a publicly accessible URL."""
    file_metadata = {'name': os.path.basename(local_path)}
    media = MediaFileUpload(local_path, mimetype='image/jpeg')

    uploaded = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, webContentLink',
    ).execute()

    file_id = uploaded['id']
    drive_service.permissions().create(
        fileId=file_id,
        body={'type': 'anyone', 'role': 'reader'},
    ).execute()

    return uploaded['webContentLink']


def clear_form(forms_service, form_id):
    """Delete every existing item from the form."""
    form = forms_service.forms().get(formId=form_id).execute()
    items = form.get('items', [])
    if not items:
        return
    requests = [
        {'deleteItem': {'location': {'index': i}}}
        for i in range(len(items) - 1, -1, -1)
    ]
    forms_service.forms().batchUpdate(
        formId=form_id, body={'requests': requests}
    ).execute()


def build_form_requests(tutorial_url, image_urls):
    """Build the list of batchUpdate requests: tutorial page, then one page per image."""
    requests = []
    idx = 0

    # Tutorial page (first page, no page break needed)
    requests.append({
        'createItem': {
            'item': {
                'imageItem': {
                    'image': {'sourceUri': tutorial_url},
                }
            },
            'location': {'index': idx},
        }
    })
    idx += 1

    for page_num, url in enumerate(image_urls, start=1):
        requests.append({
            'createItem': {
                'item': {'title': f'Question {page_num}', 'pageBreakItem': {}},
                'location': {'index': idx},
            }
        })
        idx += 1

        requests.append({
            'createItem': {
                'item': {
                    'imageItem': {
                        'image': {'sourceUri': url},
                    }
                },
                'location': {'index': idx},
            }
        })
        idx += 1

        for question_text in QUESTIONS:
            requests.append({
                'createItem': {
                    'item': {
                        'title': question_text,
                        'questionItem': {
                            'question': {
                                'required': True,
                                'choiceQuestion': {
                                    'type': 'RADIO',
                                    'options': CHOICES,
                                },
                            }
                        },
                    },
                    'location': {'index': idx},
                }
            })
            idx += 1

    return requests


def main():
    images = discover_images(IMAGE_DIR)
    if not images:
        print(f'No question*.jpg files found in {IMAGE_DIR}/')
        return
    print(f'Found {len(images)} images: {[os.path.basename(p) for p in images]}')

    creds = get_credentials()
    forms_service = build('forms', 'v1', credentials=creds)
    drive_service = build('drive', 'v3', credentials=creds)

    tutorial_path = os.path.join(IMAGE_DIR, 'tutorial.jpg')
    if not os.path.exists(tutorial_path):
        print(f'Tutorial image not found: {tutorial_path}')
        return

    print('Uploading images to Google Drive...')
    tutorial_url = upload_to_drive(drive_service, tutorial_path)
    print(f'  Uploaded tutorial.jpg')

    image_urls = []
    for path in images:
        url = upload_to_drive(drive_service, path)
        print(f'  Uploaded {os.path.basename(path)}')
        image_urls.append(url)

    print('Clearing existing form items...')
    clear_form(forms_service, FORM_ID)

    print('Building new form pages...')
    requests = build_form_requests(tutorial_url, image_urls)
    forms_service.forms().batchUpdate(
        formId=FORM_ID, body={'requests': requests}
    ).execute()

    print(f'Done! Form updated: https://docs.google.com/forms/d/{FORM_ID}')


if __name__ == '__main__':
    main()
