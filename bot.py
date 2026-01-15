"""
MISIS FAQ Telegram Bot
======================

Telegram bot for MISIS university freshmen with semantic FAQ search.

Requirements:
    pip install aiogram python-dotenv sentence-transformers faiss-cpu numpy

First run:
    1. Create .env file with BOT_TOKEN=your_token
    2. Run: python build_index.py  (builds embeddings index)
    3. Run: python bot.py
"""

import asyncio
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional

from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    Message,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardRemove
)
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage

import config
from faq_embeddings_db import FAQEmbeddingsDB, SearchResult


# ========== STATES ==========

class BotStates(StatesGroup):
    faq_mode = State()
    dialog_mode = State()


# ========== KEYBOARDS ==========

def get_main_keyboard() -> ReplyKeyboardMarkup:
    """Main menu keyboard"""
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📚 FAQ-режим")],
            [KeyboardButton(text="💬 Диалоговый режим")],
            [KeyboardButton(text="ℹ️ Помощь")]
        ],
        resize_keyboard=True
    )


def get_faq_blocks_keyboard(faq_db: FAQEmbeddingsDB) -> ReplyKeyboardMarkup:
    """Keyboard with FAQ category blocks"""
    blocks = faq_db.get_blocks()
    
    keyboard = []
    for block in sorted(blocks):
        keyboard.append([KeyboardButton(text=f"📁 {block}")])
    
    keyboard.append([KeyboardButton(text="🔍 Поиск по FAQ")])
    keyboard.append([KeyboardButton(text="🏠 В главное меню")])
    
    return ReplyKeyboardMarkup(keyboard=keyboard, resize_keyboard=True)


def get_faq_questions_keyboard(questions: List[Dict]) -> ReplyKeyboardMarkup:
    """Keyboard with questions from selected block"""
    keyboard = []
    
    for q in questions[:15]:  
        keyboard.append([KeyboardButton(text=f"❓ {q['question']}")])
    
    keyboard.append([KeyboardButton(text="⬅️ К категориям")])
    keyboard.append([KeyboardButton(text="🏠 В главное меню")])
    
    return ReplyKeyboardMarkup(keyboard=keyboard, resize_keyboard=True)


def get_search_keyboard() -> ReplyKeyboardMarkup:
    """Keyboard for search mode"""
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="⬅️ К категориям")],
            [KeyboardButton(text="🏠 В главное меню")]
        ],
        resize_keyboard=True
    )


def get_feedback_keyboard() -> InlineKeyboardMarkup:
    """Feedback buttons"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="👍", callback_data="feedback_good"),
                InlineKeyboardButton(text="👎", callback_data="feedback_bad")
            ]
        ]
    )


# ========== MAIN ==========

async def main():
    """Main bot function"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if not config.BOT_TOKEN:
        logging.error("BOT_TOKEN not found! Create .env file with BOT_TOKEN=your_token")
        return
    
    bot = Bot(token=config.BOT_TOKEN)
    dp = Dispatcher(storage=MemoryStorage())
    
    faq_db = FAQEmbeddingsDB(config.FAQ_JSON_PATH)
    
    index_path = Path(config.FAQ_INDEX_PATH)
    if index_path.with_suffix(".index").exists():
        faq_db.load(config.FAQ_INDEX_PATH)
        logging.info("Loaded existing FAQ index")
    else:
        logging.info("Building FAQ index (first run)...")
        faq_db.build_index()
        faq_db.save(config.FAQ_INDEX_PATH)
        logging.info("FAQ index built and saved")
    
    logging.info(f"FAQ database ready: {len(faq_db.items)} questions")
    
    Path("data").mkdir(exist_ok=True)
    
    user_current_block: Dict[int, str] = {}

    # ========== HANDLERS ==========

    @dp.message(F.text == "🏠 В главное меню")
    async def handle_main_menu(message: Message, state: FSMContext):
        """Return to main menu"""
        await state.clear()
        user_current_block.pop(message.from_user.id, None)
        
        await message.answer(
            "🏠 *Главное меню*\n\nВыберите режим работы:",
            parse_mode="Markdown",
            reply_markup=get_main_keyboard()
        )

    @dp.message(Command("start"))
    async def cmd_start(message: Message, state: FSMContext):
        """Handle /start command"""
        await state.clear()
        
        welcome = (
            "👋 *Добро пожаловать, первокурсник МИСИС!*\n\n"
            "Я помогу вам разобраться с учебой и жизнью в университете.\n\n"
            "*Выберите режим:*\n"
            "📚 *FAQ-режим* — быстрые ответы на частые вопросы\n"
            "💬 *Диалоговый режим* — задайте любой вопрос\n\n"
            f"В базе FAQ: *{len(faq_db.items)}* готовых ответов"
        )
        await message.answer(welcome, parse_mode="Markdown", reply_markup=get_main_keyboard())

    @dp.message(Command("help"))
    async def cmd_help(message: Message):
        """Handle /help command"""
        help_text = (
            "*Как пользоваться ботом:*\n\n"
            "*📚 FAQ-режим* — выберите категорию и вопрос из списка, "
            "или воспользуйтесь поиском.\n\n"
            "*💬 Диалоговый режим* — задавайте вопросы в свободной форме, "
            "бот найдет похожие ответы в базе.\n\n"
            "*Команды:*\n"
            "/start — главное меню\n"
            "/faq — FAQ режим\n"
            "/dialog — диалоговый режим\n"
            "/help — эта справка"
        )
        await message.answer(help_text, parse_mode="Markdown")

    @dp.message(F.text == "ℹ️ Помощь")
    async def btn_help(message: Message):
        """Help button"""
        await cmd_help(message)

    # ========== FAQ MODE ==========

    @dp.message(Command("faq"))
    @dp.message(F.text == "📚 FAQ-режим")
    async def enter_faq_mode(message: Message, state: FSMContext):
        """Enter FAQ mode - show categories"""
        await state.set_state(BotStates.faq_mode)
        
        blocks = faq_db.get_blocks()
        text = (
            "📚 *FAQ — Частые вопросы*\n\n"
            f"Выберите категорию ({len(blocks)} категорий, {len(faq_db.items)} вопросов):"
        )
        
        await message.answer(
            text,
            parse_mode="Markdown",
            reply_markup=get_faq_blocks_keyboard(faq_db)
        )

    @dp.message(F.text.startswith("📁 "), BotStates.faq_mode)
    async def select_faq_block(message: Message):
        """Select FAQ category block"""
        block_name = message.text[2:].strip()
        user_current_block[message.from_user.id] = block_name
        
        all_questions = faq_db.get_all_questions()
        block_questions = [q for q in all_questions if q["block"] == block_name]
        
        if not block_questions:
            await message.answer("В этой категории пока нет вопросов.")
            return
        
        text = f"📁 *{block_name}*\n\nВыберите вопрос ({len(block_questions)}):"
        
        await message.answer(
            text,
            parse_mode="Markdown",
            reply_markup=get_faq_questions_keyboard(block_questions)
        )

    @dp.message(F.text == "⬅️ К категориям", BotStates.faq_mode)
    async def back_to_blocks(message: Message, state: FSMContext):
        """Back to category list"""
        user_current_block.pop(message.from_user.id, None)
        await enter_faq_mode(message, state)

    @dp.message(F.text.startswith("❓ "), BotStates.faq_mode)
    async def handle_faq_question(message: Message):
        """Handle FAQ question selection"""
        question_text = message.text[2:].strip()
        
        results = faq_db.search(question_text, top_k=1)
        
        if results and results[0].score > 0.7:
            item = results[0].item
            response = (
                f"*❓ {item.question}*\n\n"
                f"📝 {item.answer}\n\n"
                f"📁 _{item.block}_"
            )
        else:
            response = (
                f"*❓ {question_text}*\n\n"
                "⚠️ Ответ не найден в базе.\n\n"
                "Попробуйте воспользоваться поиском или диалоговым режимом."
            )
        
        await message.answer(response, parse_mode="Markdown")

    @dp.message(F.text == "🔍 Поиск по FAQ", BotStates.faq_mode)
    async def enter_search_mode(message: Message):
        """Enter search mode"""
        text = (
            "🔍 *Поиск по FAQ*\n\n"
            "Введите ваш вопрос или ключевые слова, "
            "и я найду похожие вопросы в базе.\n\n"
            "Примеры:\n"
            "• когда сессия\n"
            "• как получить стипендию\n"
            "• общежитие оплата"
        )
        await message.answer(text, parse_mode="Markdown", reply_markup=get_search_keyboard())

    @dp.message(BotStates.faq_mode)
    async def handle_faq_search(message: Message):
        """Handle search query in FAQ mode"""
        query = message.text
        
        if query in ["🏠 В главное меню", "⬅️ К категориям", "🔍 Поиск по FAQ"]:
            return
        
        results = faq_db.search(query, top_k=5, score_threshold=0.3)
        
        if results:
            response = f"🔍 *Результаты поиска по «{query}»:*\n\n"
            
            for i, r in enumerate(results, 1):
                score_bar = "🟢" if r.score > 0.7 else "🟡" if r.score > 0.5 else "🔴"
                response += f"{i}. {score_bar} *{r.item.question}*\n"
                response += f"   _{r.item.answer[:100]}{'...' if len(r.item.answer) > 100 else ''}_\n\n"
            
            response += "Нажмите на вопрос в меню или введите новый запрос."
        else:
            response = (
                f"🔍 По запросу «{query}» ничего не найдено.\n\n"
                "Попробуйте:\n"
                "• Переформулировать запрос\n"
                "• Использовать другие ключевые слова\n"
                "• Перейти в диалоговый режим"
            )
        
        await message.answer(response, parse_mode="Markdown", reply_markup=get_search_keyboard())

    # ========== DIALOG MODE ==========

    @dp.message(Command("dialog"))
    @dp.message(F.text == "💬 Диалоговый режим")
    async def enter_dialog_mode(message: Message, state: FSMContext):
        """Enter dialog mode"""
        await state.set_state(BotStates.dialog_mode)
        
        text = (
            "💬 *Диалоговый режим*\n\n"
            "Задайте любой вопрос о МИСИС в свободной форме.\n\n"
            "Я поищу похожие вопросы в базе FAQ и постараюсь помочь!\n\n"
            "Примеры:\n"
            "• Как найти столовую?\n"
            "• Что делать если потерял студенческий?\n"
            "• Когда будет день открытых дверей?"
        )
        
        keyboard = ReplyKeyboardMarkup(
            keyboard=[[KeyboardButton(text="🏠 В главное меню")]],
            resize_keyboard=True
        )
        
        await message.answer(text, parse_mode="Markdown", reply_markup=keyboard)

    @dp.message(BotStates.dialog_mode)
    async def handle_dialog_question(message: Message):
        """Handle question in dialog mode"""
        query = message.text
        
        if query == "🏠 В главное меню":
            return
        
        results = faq_db.search(query, top_k=3, score_threshold=0.3)
        
        if results:
            best = results[0]
            
            if best.score > 0.75:
                response = (
                    f"💬 *На ваш вопрос:*\n_{query}_\n\n"
                    f"✅ *Нашёл ответ:*\n{best.item.answer}\n\n"
                    f"📁 _{best.item.block}_"
                )
            else:
                response = f"💬 *На ваш вопрос:*\n_{query}_\n\n"
                response += "🤔 *Возможно, вам подойдут эти ответы:*\n\n"
                
                for i, r in enumerate(results, 1):
                    response += f"{i}. *{r.item.question}*\n"
                    response += f"   {r.item.answer[:150]}{'...' if len(r.item.answer) > 150 else ''}\n\n"
        else:
            responses = [
                f"По вашему вопросу *«{query}»* я не нашёл точного ответа в базе FAQ.\n\n"
                "Рекомендую обратиться в учебную часть или деканат вашего факультета.",
                
                f"К сожалению, по запросу *«{query}»* в базе нет информации.\n\n"
                "Попробуйте переформулировать вопрос или проверьте раздел FAQ.",
            ]
            response = random.choice(responses)
        
        keyboard = ReplyKeyboardMarkup(
            keyboard=[[KeyboardButton(text="🏠 В главное меню")]],
            resize_keyboard=True
        )
        
        await message.answer(
            response,
            parse_mode="Markdown",
            reply_markup=keyboard
        )

    # ========== FEEDBACK ==========

    @dp.callback_query(F.data.startswith("feedback_"))
    async def handle_feedback(callback_query):
        """Handle feedback buttons"""
        feedback = callback_query.data.replace("feedback_", "")
        
        if feedback == "good":
            await callback_query.answer("Спасибо за оценку! 👍")
        else:
            await callback_query.answer("Учтём ваше мнение! 👎")
        
        await callback_query.message.edit_reply_markup(reply_markup=None)

    # ========== FALLBACK ==========

    @dp.message()
    async def handle_other(message: Message):
        """Handle unrecognized messages"""
        await message.answer(
            "Используйте кнопки меню или команды:\n"
            "/start — главное меню\n"
            "/faq — FAQ режим\n"
            "/dialog — диалоговый режим",
            reply_markup=get_main_keyboard()
        )

    # ========== RUN ==========

    try:
        await bot.delete_webhook(drop_pending_updates=True)
        logging.info("Bot started! Waiting for messages...")
        await dp.start_polling(bot)
    except Exception as e:
        logging.error(f"Bot error: {e}")
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
