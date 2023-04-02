import asyncio
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp

from constants.constants import *

from pprint import pprint

class ProxyCollection:
    def __init__(self) -> None:
        self.index = 0
        # Note: lengthes of list of proxies and emails must equal
        # len(self.emails) == len(self.proxies)
        self.proxies = PROXIES
        self.emails = [
            'nelson@amigoscode.com',
            'k.nordin@retool.com',
            'mabdina@scalablesolutions.io',
            'mariel.avalos@inallmedia.com',
            'Bakhtiyar.Mamyrakhunov@infobip.com'
        ]

    def get(self) -> str:
        index = self.index % len(self.emails)
        email = self.emails[index]
        proxy = self.proxies[index]

        self.index = (self.index + 1) % len(self.emails)
        return email, proxy


proxy_store = ProxyCollection()


class TranslatorBot:
    def __init__(self, from_lang, to_lang:str, use_proxy: bool = True):
        self.use_proxy = use_proxy
        self.to_lang = to_lang
        self.from_lang = from_lang
        self.chunk_size = 499

        print(f"[TranslatorBot] created a TranslatorBot with a field: {self.use_proxy}")

    async def translate(self, text):
        async with aiohttp.ClientSession() as session:
            
            email, proxy = proxy_store.get()
            proxy_url = proxy[2]
            username = proxy[0]
            password = proxy[1]

            params = {
                'q': text,
                'langpair': f"{self.from_lang}|{self.to_lang}",
            }

            if self.use_proxy:
                params['de'] = email

            if self.use_proxy:
                async with session.get(
                    'https://api.mymemory.translated.net/get', 
                    params=params,
                    proxy=proxy_url, 
                    proxy_auth=aiohttp.BasicAuth(username, password),
                ) as resp:
                    result = await resp.json()
                    if resp is None:
                        pprint(resp)
                        return 'some error has happend'
                    return result.get('responseData', dict()).get('translatedText', '')
            else:
                async with session.get('https://api.mymemory.translated.net/get', params=params) as resp:
                    result = await resp.json()
                    if resp is None:
                        pprint(resp)
                        return 'some error has happend'
                    return result.get('responseData', dict()).get('translatedText', '')

    def split_text(self, text: str, max_chunk_size: int = 499):
        sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', text)
        chunks = []
        current_chunk = ""
        current_count = 0
        for sentence in sentences:
            if '\n' in sentence:
                sentence = sentence.replace('\n', '<b>')
                sentences_split_by_newline = sentence.split('\n')
                for new_sentence in sentences_split_by_newline:
                    if current_count + len(new_sentence) <= max_chunk_size:
                        current_chunk += new_sentence
                        current_count += len(new_sentence)
                    else:
                        chunks.append(current_chunk)
                        current_chunk = new_sentence
                        current_count = len(new_sentence)
            else:
                if current_count + len(sentence) <= max_chunk_size:
                    current_chunk += sentence
                    current_count += len(sentence)
                else:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                    current_count = len(sentence)
        chunks.append(current_chunk)
        return chunks

    async def wrapper(self, text: str) -> str:
        chunks = [chunk for chunk in self.split_text(text) if len(chunk) > 0]
        tasks = [asyncio.create_task(self.translate(chunk)) for chunk in chunks]
        result = await asyncio.gather(*tasks)
        result = ''.join(result)
        result = result.replace('<b>', '\n').replace('</b>', '')

        return result

async def run():
    translator = TranslatorBot('en','kk')
    # english_text = "<b>The meaning of life is a complex and highly debated topic that has been explored by philosophers, religious figures, and scientists throughout history. </b>  <b> Some believe that the meaning of life is to seek happiness, others believe it is to fulfill a specific purpose or destiny, and still others believe that life has no inherent meaning and that it is up to each individual to create their own purpose. </b> <b>Ultimately, the meaning of life may be different for each person and can change throughout one's lifetime. </b> <b> Some might find meaning in helping others, others in achieving personal goals, some in finding inner peace, and others in pursuing spiritual enlightenment. </b> <b> The meaning of life is something that is constantly being explored and redefined by each individual as they navigate their own journey through life. </b>"
    # english_text = "The meaning of life is a complex and highly debated topic that has been explored by philosophers, religious figures, and scientists throughout history. <b> Some believe that the meaning of life is to seek happiness, others believe it is to fulfill a specific purpose or destiny, and still others believe that life has no inherent meaning and that it is up to each individual to create their own purpose. <b> Ultimately, the meaning of life may be different for each person and can change throughout one's lifetime. <b> Some might find meaning in helping others, others in achieving personal goals, some in finding inner peace, and others in pursuing spiritual enlightenment. <b> The meaning of life is something that is constantly being explored and redefined by each individual as they navigate their own journey through life. <b>"
    # english_text = '''The meaning of life is a complex and highly debated topic that has been explored by philosophers, religious figures, and scientists throughout history.\n Some believe that the meaning of life is to seek happiness, others believe it is to fulfill a specific purpose or destiny, and still others believe that life has no inherent meaning and that it is up to each individual to create their own purpose.\n Ultimately, the meaning of life may be different for each person and can change throughout one's lifetime. \n Some might find meaning in helping others, others in achieving personal goals, some in finding inner peace, and others in pursuing spiritual enlightenment. \n The meaning of life is something that is constantly being explored and redefined by each individual as they navigate their own journey through life. \n'''
    english_text = '''As a language model, I don't have personal opinions or preferences, but here is a list of some of the most highly rated and popular movies, according to various sources:

    The Shawshank Redemption (1994)
    The Godfather (1972)
    The Godfather: Part II (1974)
    The Dark Knight (2008)
    12 Angry Men (1957)
    Schindler's List (1993)
    The Lord of the Rings: The Return of the King (2003)
    Pulp Fiction (1994)
    The Good, the Bad and the Ugly (1966)
    Forrest Gump (1994)
    '''
    tasks = [asyncio.create_task(translator.wrapper(english_text))]
    result = await asyncio.gather(*tasks)
    for each in result:
        print(each)
    print(len(result))

# loop = asyncio.new_event_loop()
# asyncio.set_event_loop(loop)
# loop.run_until_complete(run())
# loop.close()

if __name__ == "__main__":
    asyncio.run(run())
