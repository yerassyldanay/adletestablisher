from telegram import InlineKeyboardButton, InlineKeyboardMarkup


class STATE:
    QUESTION, CHOOSE_LANGUAGE = range(2)


class ParseMode:
    MarkdownV2 = "MarkdownV2"
    HTML = "HTML"

class Keyboard:
    LANG = InlineKeyboardMarkup.from_row([
        InlineKeyboardButton('қазақша', callback_data='choose_language_kaz'),
        InlineKeyboardButton('русский', callback_data='choose_language_rus'),
    ])
