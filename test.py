# import requests
# response = requests.post(url='http://speaker.is99kdf.xyz:16080/pre',
#                          files={'wav': open('test_dir/wav_addnoise_snr-5+5/S0003/BAC009S0003W0121.wav', 'rb')})

# with open('tmp.wav', 'wb') as f:
#   f.write(response.content)
'''
共享内存
multiprocess组件Lock
'''
import multiprocessing
import time
from multiprocessing import Value,Array,Manager
'''
Array数组
Manager
'''

def add1(value,number):
    print("start add1 number={0}".format(number.value))
    for i in range(1,5):
        number.value += value
        print("number add1 = {0}".format(number.value))

def add3(value,number):
    print("start add3 number={0}".format(number.value))
    try:
        for i in range(1,5):
            number.value += value
            print("number add3 = {0}".format(number.value))
    except Exception as e:
        raise e

if __name__ == '__main__':
    print("start main")
    number = Value('d',0) #共用的内存地址
    p1 = multiprocessing.Process(target=add1, args=(1,number))
    p3 = multiprocessing.Process(target=add3, args=(3,number))
    p1.start()
    p3.start()
    print("end main")
