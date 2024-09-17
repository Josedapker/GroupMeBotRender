import requests

def send_to_groupme(message):
    url = 'https://api.groupme.com/v3/bots/post'
    data = {
        'bot_id': ,
        'text': message
    }
    response = requests.post(url, json=data)
    if response.status_code != 202:
        print(f'Failed to send message to GroupMe: {response.status_code}, {response.text}')
