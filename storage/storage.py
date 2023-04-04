from telegram import Update, Bot
from telegram.ext import (
    ApplicationBuilder, 
    ContextTypes, 
    CommandHandler, 
)

from constants.constants import *
from pprint import pprint

import asyncio


class FeedbackStorage:
    def __init__(self) -> None:
        self.bot = self.bot = Bot(TELEGRAM_BOT_STORAGE_TOKEN)
        self.application = ApplicationBuilder().token(TELEGRAM_BOT_STORAGE_TOKEN).build()
        self.chat_ids = [
            '420451657',
            '420353277'
        ]

        self.application.add_handler(CommandHandler("start", self.start))

    async def send(self, text: str):
        for chat_id in self.chat_ids:
            await self.bot.send_message(
                chat_id=chat_id, 
                text=text,
                parse_mode="HTML"
            )

    async def sendKeyValues(self, dictionary: dict, additional: dict = {}):
        result = []
        for key, value in dictionary.items():
            result.append(f'''{key}: {value}
''')
        for key, value in additional.items():
            result.append(f'''{key}: {value}
''')
        
        result = ''.join(result)
        for chat_id in self.chat_ids:
            await self.bot.send_message(
                chat_id=chat_id, 
                text=result,
                parse_mode="HTML"
            )

    def run(self):
        self.application.run_polling()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        pprint(update)

    async def __close(self):
        return asyncio.gather(self.application.shutdown())

    def close(self):
        asyncio.run(self.__close())


if __name__ == '__main__':
    app = FeedbackStorage()
    try:
        app.run()
    except Exception as ex:
        print(ex)
    finally:
        app.close()
