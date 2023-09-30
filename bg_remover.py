import requests
import cv2
import os


def remover_bg_fun(image):

    response = requests.post(
        'https://api.remove.bg/v1.0/removebg',
        files={'image_file': open(image,'rb')},
        data={'size': 'auto'},
        headers={'X-Api-Key': 'a2TdtHWXZBAA9xMfvNwoRAgZ'}, #a2TdtHWXZBAA9xMfvNwoRAgZ #Mkpvf7ZjfG1zREGRWWGTVMpj
    )

    name = image
    img_path = f"remove_bg/{name}"
    name = os.path.split(img_path)[-1]

    if response.status_code == requests.codes.ok:
        with open(f'remove_bg/{name}', 'wb') as out:
            img = out.write(response.content)
    else:
        print("Error:", response.status_code, response.text)

    final_img = cv2.imread(f'remove_bg/{name}')
    return final_img


if __name__ == '__main__':
    remove_bg_fun()