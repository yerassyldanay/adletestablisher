import asyncio
from telegram import Update
from telegram.ext import (
    ContextTypes, 
    PicklePersistence,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    filters,
)

from constants.constants import *

from models.monitor import UserInfo
from models.keyboards import Keyboard, ParseMode, STATE
from models.adlet import Adlet

from translator.translator import TranslatorBot

KAZ = 'kk'
RUS = 'ru'

class AdletEstablisherBot(UserInfo):
    def __init__(self) -> None:
        UserInfo.__init__(self)
        self.adlet = Adlet()
        self.__persistence = PicklePersistence(filepath="arbitrarycallbackdatabot")
        self.application = ApplicationBuilder().persistence(self.__persistence).token(TELEGRAM_BOT_TOKEN).concurrent_updates(True).build()
        
        self.__set_buttons__()

    def __set_buttons__(self):
        commandStart = CommandHandler("start", self.start)
        messageQuestion = MessageHandler(filters.TEXT, self.answer)
        callbackChooseLanguage = CallbackQueryHandler(self.change_language, pattern='choose_language_*')    

        conversation = ConversationHandler(
            entry_points=[
                commandStart,
                messageQuestion,
                callbackChooseLanguage,
            ],
            states={
                STATE.CHOOSE_LANGUAGE: [
                    commandStart,
                    callbackChooseLanguage,
                ],
                STATE.QUESTION: [
                    commandStart,
                    messageQuestion,
                    callbackChooseLanguage,
                ],
            },
            fallbacks=[
                commandStart,
                messageQuestion,
                callbackChooseLanguage,
            ]
        )
        self.application.add_handler(conversation)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            parse_mode=ParseMode.HTML, 
            text="Тілді таңдаңыз / Выберите язык",
            reply_markup=Keyboard.LANG,
        )

        user_info = self.get_user_info(update=update)
        self.log(user_info, { "COMMAND": "/start" })

        context.user_data["language"] = KAZ
        return STATE.CHOOSE_LANGUAGE
    
    async def change_language(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user_info = self.get_user_info(update=update)

        query = update.callback_query
        language_chosen = query.data
        
        self.log(user_info, { "command": "change_language", "lang": language_chosen})

        text = "Өз сұрағыңызды қойыңыз"
        context.user_data["language"] = KAZ
        if language_chosen == "choose_language_rus":
            context.user_data["language"] = RUS
            text = "Задавайте свой вопрос"

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=text,
            parse_mode=ParseMode.HTML
        )

        return STATE.QUESTION

    @staticmethod
    def get_translator(from_lang, to_lang: str):
        return TranslatorBot(from_lang=from_lang, to_lang=to_lang, use_proxy=False)

    async def answer(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_info = self.get_user_info(update)

        question = update.message.text
        self.log(user_info, { "command": "answer", "type": "user_input", "question": question})

        # translate the message
        lang = context.user_data["language"]
        question = await self.get_translator(lang, "en").wrapper(question)
        self.log(user_info, { "command": "answer", "type": "translator_1", "question": question})

        # ask the question from the GPT-trained model
        answer = self.adlet.answer(question)
        self.log(user_info, { "command": "answer", "type": "gpt_model", "answer": answer})

        # translate from English to the original language
        answer = await self.get_translator("en", lang).wrapper(answer)
        self.log(user_info, { "command": "answer", "type": "translator_2", "answer": answer})

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=answer,
            parse_mode=ParseMode.HTML
        )
        return STATE.QUESTION

    def run(self):
        self.application.run_polling()

    async def __close(self):
        return asyncio.gather(self.application.shutdown())

    def close(self):
        asyncio.run(self.__close())


if __name__ == '__main__':
    app = AdletEstablisherBot()
    try:
        app.run()
    except Exception as ex:
        print(ex)
    finally:
        app.close()
