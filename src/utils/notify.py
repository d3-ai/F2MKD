import requests
def Line(message, token="8ca0leockIJ3dZTH5ZmuMDZkgzfqtr992dgfxtKFAbH"):
    """
    LINEにmessageを通知
    """
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": "Bearer " + token}

    payload = {"message":  message}

    r = requests.post(url,headers=headers,params=payload)
    print(r)