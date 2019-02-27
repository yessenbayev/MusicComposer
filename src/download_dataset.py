import requests 
import os
import re
import time
from bs4 import BeautifulSoup 
  
''' 
URL of the archive web-page which provides link to 
all video lectures. It would have been tiring to 
download each video manually. 
In this example, we first crawl the webpage to extract 
all the links and then download videos. 
'''
  
# specify the URL of the archive here 
archive_url = "https://www.vgmusic.com"

def needed_links(href):
    return href and re.compile("/music/").search(href)

def get_audio_links(): 


      
    # create response object 
    r = requests.get(archive_url) 
      
    # create beautiful-soup object 
    soup = BeautifulSoup(r.content,'html5lib') 
      
    # find all links on web-page 
    links = soup.findAll('a',href=needed_links) 
    audio_links = []

    for link in links:
        r2 = requests.get(archive_url+link['href'].strip('.')) 
        soup = BeautifulSoup(r2.content,'html5lib') 
        download_links = soup.findAll('a',href=True) 
    # filter the link sending with .mid
        audio_links += [archive_url + link['href'].strip('.') + download_link['href'] for download_link in download_links if download_link['href'].endswith('mid')] 
    return audio_links 
  
  
def download_audio_series(audio_links,directory): 
  
    for link in audio_links: 
        try:
            '''iterate through all links in audio_links 
            and download them one by one'''
                
            # obtain filename by splitting url and getting  
            # last string 
            file_name = link.split('/')[-1]
            file_system = link.split('/')[-2]

            if not os.path.exists(os.path.join(directory,file_system)):
                os.makedirs(os.path.join(directory,file_system))

            print("Downloading file:%s"%file_name)

            # create response object 
            r = requests.get(link, stream = True) 
                
            # download started 
            with open(os.path.join(directory,file_system,file_name), 'wb') as f: 
                for chunk in r.iter_content(chunk_size = 1024*1024): 
                    if chunk: 
                        f.write(chunk) 
                
            print("%s downloaded!\n"%file_name) 
        except:
            time.sleep(1)
            with open('log.txt','a') as f:
                f.write(link + ' failed\n')
  
    print("All audios downloaded!")
    return
  
  
if __name__ == "__main__": 
    directory = 'data'
    if not os.path.exists(directory):
        os.makedirs(directory)
  
    # getting all video links 
    print("Finding all links on the website")
    audio_links = get_audio_links() 
  
    # download all videos 
    download_audio_series(audio_links,directory) 