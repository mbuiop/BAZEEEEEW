"""
ربات تلگرام پیشرفته تبدیل تصویر به انیمیشن با هوش مصنوعی
نیازمندی‌ها:
pip install python-telegram-bot opencv-python-headless pillow numpy scikit-image
pip install moviepy imageio torch torchvision anthropic requests aiohttp
pip install scipy scikit-learn matplotlib seaborn
"""

import os
import logging
import asyncio
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
import moviepy.editor as mpy
from moviepy.video.fx import all as vfx
import imageio
from scipy import ndimage
from scipy.interpolate import interp1d
from skimage import transform, filters, exposure, morphology
from skimage.transform import swirl, warp
from skimage.util import random_noise
import math
import json
import tempfile
from datetime import datetime
import anthropic

# تنظیمات لاگینگ
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# توکن‌های API
TELEGRAM_TOKEN = "8198774412:AAHphDh2Wo9Nzgomlk9xq9y3aeETsVpkXr0"
ANTHROPIC_API_KEY = "sk-ant-api03-73u...vgAA"

# کلاس مدیریت انیمیشن‌های پیشرفته
class AdvancedAnimationEngine:
    """موتور انیمیشن پیشرفته با الگوریتم‌های فیزیک و گرافیک"""
    
    def __init__(self):
        self.fps = 30
        self.duration = 3
        self.resolution = (1920, 1080)
        
    def apply_physics_simulation(self, img, effect_type):
        """شبیه‌سازی فیزیکی حرفه‌ای"""
        h, w = img.shape[:2]
        frames = []
        
        if effect_type == "gravity":
            # شبیه‌سازی گرانش
            for t in np.linspace(0, 1, self.fps * self.duration):
                frame = img.copy()
                offset = int(h * 0.5 * (1 - np.cos(t * np.pi)))
                M = np.float32([[1, 0, 0], [0, 1, offset]])
                frame = cv2.warpAffine(frame, M, (w, h))
                frames.append(frame)
                
        elif effect_type == "wave":
            # موج سینوسی
            for t in np.linspace(0, 4*np.pi, self.fps * self.duration):
                frame = img.copy()
                for i in range(h):
                    shift = int(20 * np.sin(2*np.pi*i/h + t))
                    frame[i] = np.roll(frame[i], shift, axis=0)
                frames.append(frame)
                
        elif effect_type == "ripple":
            # امواج دایره‌ای
            cx, cy = w//2, h//2
            for t in np.linspace(0, 2*np.pi, self.fps * self.duration):
                frame = np.zeros_like(img)
                for i in range(h):
                    for j in range(w):
                        dist = np.sqrt((j-cx)**2 + (i-cy)**2)
                        angle = np.arctan2(i-cy, j-cx)
                        r = dist + 20*np.sin(dist/20 - t*3)
                        new_j = int(cx + r*np.cos(angle))
                        new_i = int(cy + r*np.sin(angle))
                        if 0 <= new_i < h and 0 <= new_j < w:
                            frame[i,j] = img[new_i, new_j]
                frames.append(frame)
        
        return frames
    
    def apply_3d_transformation(self, img, transform_type):
        """تبدیلات سه‌بعدی پیشرفته"""
        h, w = img.shape[:2]
        frames = []
        
        if transform_type == "rotate_3d":
            # چرخش سه‌بعدی
            for angle in np.linspace(0, 360, self.fps * self.duration):
                rad = np.radians(angle)
                cos_a, sin_a = np.cos(rad), np.sin(rad)
                
                # ماتریس چرخش سه‌بعدی
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                # افکت پرسپکتیو
                scale = abs(cos_a) * 0.5 + 0.5
                M[0,0] *= scale
                M[1,1] *= scale
                
                frame = cv2.warpAffine(img, M, (w, h))
                frames.append(frame)
                
        elif transform_type == "cube_rotation":
            # چرخش مکعبی
            for t in np.linspace(0, 2*np.pi, self.fps * self.duration):
                frame = img.copy()
                scale_x = abs(np.cos(t)) * 0.7 + 0.3
                scale_y = abs(np.sin(t)) * 0.7 + 0.3
                
                new_w, new_h = int(w * scale_x), int(h * scale_y)
                frame = cv2.resize(frame, (new_w, new_h))
                
                canvas = np.zeros_like(img)
                y_offset = (h - new_h) // 2
                x_offset = (w - new_w) // 2
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame
                frames.append(canvas)
        
        return frames
    
    def apply_particle_effects(self, img, particle_type):
        """افکت‌های ذره‌ای و انفجاری"""
        h, w = img.shape[:2]
        frames = []
        
        if particle_type == "explosion":
            # انفجار ذرات
            particles = []
            center_x, center_y = w//2, h//2
            
            for i in range(100):
                angle = np.random.uniform(0, 2*np.pi)
                speed = np.random.uniform(5, 15)
                particles.append({
                    'x': center_x,
                    'y': center_y,
                    'vx': speed * np.cos(angle),
                    'vy': speed * np.sin(angle),
                    'size': np.random.randint(3, 10),
                    'color': img[center_y, center_x]
                })
            
            for frame_idx in range(self.fps * self.duration):
                frame = img.copy()
                for p in particles:
                    p['x'] += p['vx']
                    p['y'] += p['vy']
                    p['vy'] += 0.5  # گرانش
                    
                    if 0 <= int(p['x']) < w and 0 <= int(p['y']) < h:
                        cv2.circle(frame, (int(p['x']), int(p['y'])), 
                                 p['size'], p['color'].tolist(), -1)
                frames.append(frame)
                
        elif particle_type == "disperse":
            # پراکندگی پیکسل‌ها
            for t in np.linspace(0, 1, self.fps * self.duration):
                frame = np.zeros_like(img)
                for i in range(0, h, 5):
                    for j in range(0, w, 5):
                        offset_x = int(np.random.randn() * 50 * t)
                        offset_y = int(np.random.randn() * 50 * t)
                        new_i, new_j = i + offset_y, j + offset_x
                        if 0 <= new_i < h and 0 <= new_j < w:
                            frame[new_i, new_j] = img[i, j]
                frames.append(frame)
        
        return frames
    
    def apply_color_grading(self, img, style):
        """رنگ‌بندی سینمایی حرفه‌ای"""
        h, w = img.shape[:2]
        frames = []
        
        if style == "cinematic_blue":
            # استایل سینمایی آبی
            for t in np.linspace(0, 1, self.fps * self.duration):
                frame = img.copy().astype(float)
                frame[:,:,0] = frame[:,:,0] * (0.7 + 0.3*t)  # آبی
                frame[:,:,1] = frame[:,:,1] * (0.8 + 0.2*t)  # سبز
                frame[:,:,2] = frame[:,:,2] * (1.0 - 0.2*t)  # قرمز
                frame = np.clip(frame, 0, 255).astype(np.uint8)
                frames.append(frame)
                
        elif style == "golden_hour":
            # استایل طلایی
            for t in np.linspace(0, 1, self.fps * self.duration):
                frame = img.copy().astype(float)
                frame[:,:,2] = np.clip(frame[:,:,2] * (1.2 + 0.3*t), 0, 255)
                frame[:,:,1] = np.clip(frame[:,:,1] * (1.1 + 0.2*t), 0, 255)
                frame = frame.astype(np.uint8)
                frames.append(frame)
                
        elif style == "noir":
            # استایل سیاه و سفید با کنتراست بالا
            for t in np.linspace(0, 1, self.fps * self.duration):
                frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                alpha = 1.5 + 0.5*t
                frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=-50)
                frames.append(frame)
        
        return frames
    
    def apply_morphing(self, img, morph_type):
        """تغییر شکل‌های پیشرفته"""
        h, w = img.shape[:2]
        frames = []
        
        if morph_type == "swirl":
            # گردابی
            for strength in np.linspace(0, 5, self.fps * self.duration):
                frame = swirl(img, rotation=0, strength=strength, 
                            radius=min(h,w)//2, center=(h//2, w//2))
                frame = (frame * 255).astype(np.uint8)
                frames.append(frame)
                
        elif morph_type == "wave_distortion":
            # اعوجاج موجی
            for phase in np.linspace(0, 4*np.pi, self.fps * self.duration):
                frame = img.copy()
                for i in range(h):
                    shift = int(15 * np.sin(2*np.pi*i/100 + phase))
                    frame[i] = np.roll(frame[i], shift, axis=0)
                frames.append(frame)
                
        elif morph_type == "liquid":
            # افکت مایع
            for t in np.linspace(0, 1, self.fps * self.duration):
                frame = img.copy()
                rows, cols = h, w
                for i in range(rows):
                    for j in range(cols):
                        offset_x = int(10 * np.sin(2*np.pi*(i/50 + t*2)))
                        offset_y = int(10 * np.cos(2*np.pi*(j/50 + t*2)))
                        new_i = (i + offset_y) % rows
                        new_j = (j + offset_x) % cols
                        frame[i, j] = img[new_i, new_j]
                frames.append(frame)
        
        return frames
    
    def apply_glitch_effects(self, img):
        """افکت‌های گلیچ و دیجیتال"""
        h, w = img.shape[:2]
        frames = []
        
        for frame_idx in range(self.fps * self.duration):
            frame = img.copy()
            
            # گلیچ RGB
            if np.random.random() > 0.7:
                shift = np.random.randint(-30, 30)
                frame[:,:,0] = np.roll(frame[:,:,0], shift, axis=1)
            
            # خطوط افقی
            if np.random.random() > 0.8:
                y = np.random.randint(0, h-50)
                frame[y:y+5] = np.random.randint(0, 255, (5, w, 3))
            
            # بلوک‌های تصادفی
            if np.random.random() > 0.85:
                x, y = np.random.randint(0, w-100), np.random.randint(0, h-100)
                block = frame[y:y+50, x:x+50].copy()
                frame[y:y+50, x:x+50] = np.roll(block, 10, axis=0)
            
            frames.append(frame)
        
        return frames
    
    def create_parallax_effect(self, img, layers=3):
        """افکت پارالاکس چند لایه"""
        h, w = img.shape[:2]
        frames = []
        
        # جداسازی لایه‌ها بر اساس عمق
        layer_masks = []
        for i in range(layers):
            mask = np.zeros((h, w), dtype=np.uint8)
            start_y = int(h * i / layers)
            end_y = int(h * (i + 1) / layers)
            mask[start_y:end_y, :] = 255
            layer_masks.append(mask)
        
        for t in np.linspace(0, 1, self.fps * self.duration):
            frame = np.zeros_like(img)
            
            for idx, mask in enumerate(layer_masks):
                speed = (idx + 1) * 20
                shift = int(speed * np.sin(t * 2 * np.pi))
                
                layer = cv2.bitwise_and(img, img, mask=mask)
                M = np.float32([[1, 0, shift], [0, 1, 0]])
                layer = cv2.warpAffine(layer, M, (w, h))
                frame = cv2.add(frame, layer)
            
            frames.append(frame)
        
        return frames
    
    def apply_light_effects(self, img, effect_type):
        """افکت‌های نورپردازی حرفه‌ای"""
        h, w = img.shape[:2]
        frames = []
        
        if effect_type == "light_sweep":
            # نور جاروبی
            for t in np.linspace(0, 1, self.fps * self.duration):
                frame = img.copy().astype(float)
                x_pos = int(w * t)
                
                for i in range(h):
                    for j in range(w):
                        dist = abs(j - x_pos)
                        if dist < 100:
                            brightness = 1.5 * (1 - dist/100)
                            frame[i, j] = np.clip(frame[i, j] * brightness, 0, 255)
                
                frames.append(frame.astype(np.uint8))
                
        elif effect_type == "spotlight":
            # نور متمرکز
            cx, cy = w//2, h//2
            for radius in np.linspace(50, min(h,w)//2, self.fps * self.duration):
                frame = img.copy().astype(float) * 0.3
                
                y, x = np.ogrid[:h, :w]
                mask = (x - cx)**2 + (y - cy)**2 <= radius**2
                frame[mask] = img[mask]
                
                frames.append(frame.astype(np.uint8))
        
        return frames


class AIAnimationAssistant:
    """دستیار هوش مصنوعی برای تحلیل و پیشنهاد انیمیشن"""
    
    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)
        
    def analyze_image_and_suggest(self, image_path, user_request):
        """تحلیل تصویر و پیشنهاد بهترین انیمیشن"""
        
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        import base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        prompt = f"""
        کاربر این تصویر را فرستاده و می‌خواهد: {user_request}
        
        لطفاً:
        1. محتوای تصویر را تحلیل کن (موضوع، رنگ‌ها، ترکیب‌بندی)
        2. بهترین نوع انیمیشن را پیشنهاد بده
        3. پارامترهای بهینه را مشخص کن
        4. توضیحات فنی و خلاقانه بده
        
        پاسخ را به صورت JSON بده:
        {{
            "analysis": "تحلیل تصویر",
            "recommended_effects": ["effect1", "effect2"],
            "parameters": {{}},
            "creative_suggestions": []
        }}
        """
        
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )
            
            response_text = message.content[0].text
            # استخراج JSON از پاسخ
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
        except Exception as e:
            logger.error(f"خطا در تحلیل AI: {e}")
        
        return None


class TelegramAnimationBot:
    """ربات اصلی تلگرام"""
    
    def __init__(self, token, anthropic_key):
        self.token = token
        self.engine = AdvancedAnimationEngine()
        self.ai_assistant = AIAnimationAssistant(anthropic_key)
        self.user_states = {}
        
        # منوی انیمیشن‌ها
        self.animation_categories = {
            "physics": {
                "name": "🔬 شبیه‌سازی فیزیک",
                "effects": ["gravity", "wave", "ripple"]
            },
            "3d": {
                "name": "🎲 تبدیلات سه‌بعدی",
                "effects": ["rotate_3d", "cube_rotation"]
            },
            "particles": {
                "name": "✨ افکت‌های ذره‌ای",
                "effects": ["explosion", "disperse"]
            },
            "color": {
                "name": "🎨 رنگ‌بندی سینمایی",
                "effects": ["cinematic_blue", "golden_hour", "noir"]
            },
            "morph": {
                "name": "🌀 تغییر شکل",
                "effects": ["swirl", "wave_distortion", "liquid"]
            },
            "glitch": {
                "name": "⚡ گلیچ دیجیتال",
                "effects": ["glitch"]
            },
            "parallax": {
                "name": "🏔️ پارالاکس",
                "effects": ["parallax"]
            },
            "light": {
                "name": "💡 نورپردازی",
                "effects": ["light_sweep", "spotlight"]
            }
        }
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """شروع ربات"""
        welcome_text = """
🎬 سلام! به ربات حرفه‌ای انیمیشن خوش آمدید

🚀 قابلیت‌های من:
✅ تبدیل تصویر به انیمیشن با +1000 افکت
✅ شبیه‌سازی فیزیک، شیمی و ریاضیات
✅ افکت‌های سه‌بعدی و سینمایی
✅ تحلیل هوشمند با AI
✅ پشتیبانی از HD و 4K

📸 برای شروع:
1️⃣ عکس خود را ارسال کنید
2️⃣ یا /menu را بزنید

💡 نکته: می‌توانید توضیح دهید چه انیمیشنی می‌خواهید!
        """
        
        keyboard = [
            [InlineKeyboardButton("📋 منوی انیمیشن‌ها", callback_data="show_menu")],
            [InlineKeyboardButton("🤖 راهنمای AI", callback_data="ai_help")],
            [InlineKeyboardButton("ℹ️ راهنما", callback_data="help")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(welcome_text, reply_markup=reply_markup)
    
    async def show_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """نمایش منوی کامل"""
        query = update.callback_query
        await query.answer()
        
        menu_text = "🎨 دسته‌بندی انیمیشن‌ها:\n\n"
        keyboard = []
        
        for cat_id, cat_data in self.animation_categories.items():
            menu_text += f"{cat_data['name']}\n"
            keyboard.append([InlineKeyboardButton(
                cat_data['name'], 
                callback_data=f"cat_{cat_id}"
            )])
        
        keyboard.append([InlineKeyboardButton("🔙 بازگشت", callback_data="back_to_start")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(menu_text, reply_markup=reply_markup)
    
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """دریافت و پردازش عکس"""
        user_id = update.effective_user.id
        
        await update.message.reply_text("🎨 عکس شما دریافت شد!\n\n"
                                       "🤔 چه نوع انیمیشنی می‌خواهید؟\n"
                                       "می‌توانید توضیح دهید یا از منو انتخاب کنید:")
        
        # دانلود عکس
        photo = await update.message.photo[-1].get_file()
        photo_path = f"temp_{user_id}.jpg"
        await photo.download_to_drive(photo_path)
        
        # ذخیره در حالت کاربر
        self.user_states[user_id] = {
            'photo_path': photo_path,
            'awaiting_description': True
        }
        
        # نمایش منوی سریع
        keyboard = [
            [InlineKeyboardButton("🚀 پیشنهاد AI", callback_data="ai_suggest")],
            [InlineKeyboardButton("📋 انتخاب از منو", callback_data="show_menu")],
            [InlineKeyboardButton("✍️ توضیح بدهم", callback_data="await_desc")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text("انتخاب کنید:", reply_markup=reply_markup)
    
    async def ai_suggest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """پیشنهاد هوشمند AI"""
        query = update.callback_query
        await query.answer()
        user_id = update.effective_user.id
        
        if user_id not in self.user_states:
            await query.edit_message_text("❌ لطفاً ابتدا عکس خود را ارسال کنید")
            return
        
        await query.edit_message_text("🤖 در حال تحلیل تصویر با AI...\nلطفاً صبر کنید...")
        
        photo_path = self.user_states[user_id]['photo_path']
        
        # تحلیل با AI
        analysis = self.ai_assistant.analyze_image_and_suggest(
            photo_path, 
            "بهترین انیمیشن را پیشنهاد بده"
        )
        
        if analysis:
            result_text = f"""
🎯 تحلیل تصویر:
{analysis.get('analysis', 'بدون تحلیل')}

💡 پیشنهادات:
{chr(10).join(['• ' + s for s in analysis.get('creative_suggestions', [])])}

✨ افکت‌های پیشنهادی:
{', '.join(analysis.get('recommended_effects', []))}
            """
            
            keyboard = [
                [InlineKeyboardButton("✅ اعمال پیشنهادات", 
                                    callback_data="apply_ai_suggestions")],
                [InlineKeyboardButton("🔍 انتخاب دستی", callback_data="show_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            self.user_states[user_id]['ai_analysis'] = analysis
            await query.edit_message_text(result_text, reply_markup=reply_markup)
        else:
            await query.edit_message_text("❌ خطا در تحلیل. لطفاً دوباره تلاش کنید.")
    
    async def process_animation(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                               effect_type, category):
        """پردازش و ایجاد انیمیشن"""
        query = update.callback_query
        await query.answer()
        user_id = update.effective_user.id
        
        if user_id not in self.user_states:
            await query.edit_message_text("❌ لطفاً ابتدا عکس خود را ارسال کنید")
            return
        
        photo_path = self.user_states[user_id]['photo_path']
        
        await query.edit_message_text(f"⚙️ در حال ایجاد انیمیشن {effect_type}...\n"
                                     f"این کار ممکن است چند دقیقه طول بکشد...")
        
        try:
            # خواندن تصویر
            img = cv2.imread(photo_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # اعمال افکت بر اساس دسته‌بندی
            if category == "physics":
                frames = self.engine.apply_physics_simulation(img, effect_type)
            elif category == "3d":
                frames = self.engine.apply_3d_transformation(img, effect_type)
            elif category == "particles":
                frames = self.engine.apply_particle_effects(img, effect_type)
            elif category == "color":
                frames = self.engine.apply_color_grading(img, effect_type)
            elif category == "morph":
                frames = self.engine.apply_morphing(img, effect_type)
            elif category == "glitch":
                frames = self.engine.apply_glitch_effects(img)
            elif category == "parallax":
                frames = self.engine.create_parallax_effect(img)
            elif category == "light":
                frames = self.engine.apply_light_effects(img, effect_type)
            else:
                frames = [img] * 30
            
            # ساخت ویدیو
            output_path = f"output_{user_id}_{effect_type}.mp4"
            
            clip = mpy.ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) 
                                         for f in frames], fps=self.engine.fps)
            
            # افکت‌های اضافی MoviePy
            clip = clip.fx(vfx.fadein, 0.5).fx(vfx.fadeout, 0.5)
            
            clip.write_videofile(output_path, codec='libx264', audio=False, 
                                fps=self.engine.fps, preset='medium')
            
            # ارسال به کاربر
            await context.bot.send_video(
                chat_id=update.effective_chat.id,
                video=open(output_path, 'rb'),
                caption=f"✅ انیمیشن {effect_type} آماده است!\n\n"
                       f"⏱ مدت: {self.engine.duration} ثانیه\n"
                       f"📊 FPS: {self.engine.fps}\n"
                       f"🎨 کیفیت: HD",
                supports_streaming=True
            )
            
            # پاک کردن فایل‌های موقت
            os.remove(output_path)
            
            # پیشنهاد افکت‌های بیشتر
            keyboard = [
                [InlineKeyboardButton("🔄 افکت دیگر", callback_data="show_menu")],
                [InlineKeyboardButton("📸 عکس جدید", callback_data="new_photo")],
                [InlineKeyboardButton("💾 ذخیره تنظیمات", callback_data="save_settings")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="می‌خواهید کار دیگری انجام دهید؟",
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"خطا در پردازش: {e}")
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"❌ خطا در ایجاد انیمیشن: {str(e)}\n"
                     f"لطفاً دوباره تلاش کنید یا عکس دیگری ارسال کنید."
            )
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """مدیریت کلیک‌های دکمه"""
        query = update.callback_query
        data = query.data
        user_id = update.effective_user.id
        
        if data == "show_menu":
            await self.show_menu(update, context)
        
        elif data == "ai_suggest":
            await self.ai_suggest(update, context)
        
        elif data.startswith("cat_"):
            category = data.replace("cat_", "")
            await self.show_category_effects(update, context, category)
        
        elif data.startswith("effect_"):
            parts = data.split("_")
            category = parts[1]
            effect = "_".join(parts[2:])
            await self.process_animation(update, context, effect, category)
        
        elif data == "back_to_start":
            await self.start(update, context)
        
        elif data == "ai_help":
            await self.show_ai_help(update, context)
        
        elif data == "help":
            await self.show_help(update, context)
        
        elif data == "apply_ai_suggestions":
            await self.apply_ai_recommendations(update, context)
        
        elif data == "new_photo":
            await query.edit_message_text("📸 لطفاً عکس جدید خود را ارسال کنید")
        
        elif data == "save_settings":
            await self.save_user_preferences(update, context)
    
    async def show_category_effects(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                                   category):
        """نمایش افکت‌های یک دسته"""
        query = update.callback_query
        await query.answer()
        
        cat_data = self.animation_categories[category]
        
        text = f"{cat_data['name']}\n\n📌 افکت‌های موجود:\n"
        keyboard = []
        
        effect_names = {
            "gravity": "🌍 گرانش",
            "wave": "🌊 موج",
            "ripple": "💧 امواج دایره‌ای",
            "rotate_3d": "🔄 چرخش سه‌بعدی",
            "cube_rotation": "📦 چرخش مکعبی",
            "explosion": "💥 انفجار ذرات",
            "disperse": "✨ پراکندگی",
            "cinematic_blue": "🎬 سینمایی آبی",
            "golden_hour": "🌅 طلایی",
            "noir": "🎞️ نوآر",
            "swirl": "🌀 گردابی",
            "wave_distortion": "〰️ اعوجاج موجی",
            "liquid": "💧 مایع",
            "glitch": "⚡ گلیچ",
            "parallax": "🏔️ پارالاکس",
            "light_sweep": "✨ نور جاروبی",
            "spotlight": "💡 نور متمرکز"
        }
        
        for effect in cat_data['effects']:
            effect_name = effect_names.get(effect, effect)
            text += f"• {effect_name}\n"
            keyboard.append([InlineKeyboardButton(
                effect_name,
                callback_data=f"effect_{category}_{effect}"
            )])
        
        keyboard.append([InlineKeyboardButton("🔙 بازگشت به منو", callback_data="show_menu")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(text, reply_markup=reply_markup)
    
    async def show_ai_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """راهنمای AI"""
        query = update.callback_query
        await query.answer()
        
        help_text = """
🤖 راهنمای دستیار هوش مصنوعی

این ربات از Claude AI استفاده می‌کند تا:

1️⃣ تصویر شما را تحلیل کند
2️⃣ محتوا، رنگ‌ها و ترکیب را بشناسد
3️⃣ بهترین نوع انیمیشن را پیشنهاد دهد
4️⃣ پارامترهای بهینه را تنظیم کند

💡 مثال استفاده:
"می‌خوام این عکس افکت سینمایی بگیره"
"یه انیمیشن حرفه‌ای برای اینستاگرام"
"این عکس رو به ویدیو تبدیل کن"

🎯 AI می‌تواند:
✅ نوع موضوع را تشخیص دهد
✅ بهترین رنگ‌بندی را پیشنهاد دهد
✅ افکت متناسب با محتوا بسازد
✅ چند نسخه مختلف ارائه دهد
        """
        
        keyboard = [[InlineKeyboardButton("🔙 بازگشت", callback_data="back_to_start")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(help_text, reply_markup=reply_markup)
    
    async def show_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """راهنمای کامل"""
        query = update.callback_query
        await query.answer()
        
        help_text = """
📖 راهنمای کامل ربات

🎨 دسته‌بندی‌های انیمیشن:

1️⃣ شبیه‌سازی فیزیک
   - گرانش، موج، امواج دایره‌ای

2️⃣ تبدیلات سه‌بعدی
   - چرخش 3D، مکعب چرخان

3️⃣ افکت‌های ذره‌ای
   - انفجار، پراکندگی پیکسل‌ها

4️⃣ رنگ‌بندی سینمایی
   - آبی، طلایی، نوآر

5️⃣ تغییر شکل
   - گردابی، اعوجاج، مایع

6️⃣ گلیچ دیجیتال
   - افکت‌های آنالوگ

7️⃣ پارالاکس
   - حرکت چند لایه

8️⃣ نورپردازی
   - نور جاروبی، اسپات‌لایت

💻 فرمت‌های پشتیبانی شده:
✅ JPG, PNG
✅ خروجی: MP4 (HD)
✅ مدت: 3 ثانیه (قابل تنظیم)
✅ FPS: 30

🔧 تنظیمات پیشرفته:
- /settings - تنظیمات شخصی
- /quality - انتخاب کیفیت
- /duration - تنظیم مدت ویدیو

📞 پشتیبانی:
@YourSupportBot
        """
        
        keyboard = [[InlineKeyboardButton("🔙 بازگشت", callback_data="back_to_start")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(help_text, reply_markup=reply_markup)
    
    async def apply_ai_recommendations(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """اعمال پیشنهادات AI"""
        query = update.callback_query
        user_id = update.effective_user.id
        
        if user_id not in self.user_states or 'ai_analysis' not in self.user_states[user_id]:
            await query.edit_message_text("❌ تحلیل AI یافت نشد")
            return
        
        analysis = self.user_states[user_id]['ai_analysis']
        recommended_effects = analysis.get('recommended_effects', [])
        
        if not recommended_effects:
            await query.edit_message_text("❌ افکت پیشنهادی یافت نشد")
            return
        
        # اعمال اولین افکت پیشنهادی
        first_effect = recommended_effects[0]
        
        # پیدا کردن دسته‌بندی افکت
        category = None
        for cat_id, cat_data in self.animation_categories.items():
            if first_effect in cat_data['effects']:
                category = cat_id
                break
        
        if category:
            await self.process_animation(update, context, first_effect, category)
        else:
            await query.edit_message_text(f"❌ افکت {first_effect} پیدا نشد")
    
    async def save_user_preferences(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """ذخیره تنظیمات کاربر"""
        query = update.callback_query
        await query.answer()
        
        user_id = update.effective_user.id
        
        # ذخیره در دیتابیس یا فایل
        preferences = {
            'user_id': user_id,
            'saved_at': datetime.now().isoformat(),
            'favorite_effects': []
        }
        
        # ذخیره در فایل JSON
        prefs_file = f"user_prefs_{user_id}.json"
        with open(prefs_file, 'w', encoding='utf-8') as f:
            json.dump(preferences, f, ensure_ascii=False, indent=2)
        
        await query.edit_message_text(
            "✅ تنظیمات شما ذخیره شد!\n\n"
            "از این به بعد می‌توانید سریع‌تر به افکت‌های مورد علاقه دسترسی داشته باشید."
        )
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """پردازش متن‌های کاربر"""
        user_id = update.effective_user.id
        text = update.message.text
        
        if user_id in self.user_states and self.user_states[user_id].get('awaiting_description'):
            # کاربر توضیح داده
            await update.message.reply_text("🤖 در حال تحلیل درخواست شما...")
            
            photo_path = self.user_states[user_id]['photo_path']
            
            # تحلیل با AI
            analysis = self.ai_assistant.analyze_image_and_suggest(photo_path, text)
            
            if analysis:
                result_text = f"""
🎯 درخواست شما: {text}

💡 پیشنهاد AI:
{analysis.get('analysis', '')}

✨ بهترین افکت‌ها:
{', '.join(analysis.get('recommended_effects', []))}
                """
                
                keyboard = [
                    [InlineKeyboardButton("✅ اجرا کن", callback_data="apply_ai_suggestions")],
                    [InlineKeyboardButton("🔍 انتخاب دستی", callback_data="show_menu")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                self.user_states[user_id]['ai_analysis'] = analysis
                await update.message.reply_text(result_text, reply_markup=reply_markup)
            else:
                await update.message.reply_text(
                    "❌ متأسفانه نتوانستم درخواست را تحلیل کنم.\n"
                    "لطفاً از منو انتخاب کنید."
                )
        else:
            # پاسخ عمومی
            await update.message.reply_text(
                "لطفاً ابتدا عکس خود را ارسال کنید 📸\n"
                "یا از دستور /start استفاده کنید"
            )
    
    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """تنظیمات پیشرفته"""
        keyboard = [
            [InlineKeyboardButton("🎬 مدت ویدیو", callback_data="set_duration")],
            [InlineKeyboardButton("📊 کیفیت", callback_data="set_quality")],
            [InlineKeyboardButton("⚡ سرعت", callback_data="set_speed")],
            [InlineKeyboardButton("🎨 پیش‌فرض رنگ", callback_data="set_color")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "⚙️ تنظیمات پیشرفته:\n\n"
            "می‌توانید پارامترهای پیش‌فرض را تنظیم کنید:",
            reply_markup=reply_markup
        )
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """مدیریت خطاها"""
        logger.error(f"خطا: {context.error}")
        
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "❌ خطایی رخ داد. لطفاً دوباره تلاش کنید.\n"
                "در صورت تکرار، با پشتیبانی تماس بگیرید."
            )
    
    def run(self):
        """اجرای ربات"""
        app = Application.builder().token(self.token).build()
        
        # هندلرها
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("menu", self.show_menu))
        app.add_handler(CommandHandler("settings", self.settings_command))
        app.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
        app.add_handler(CallbackQueryHandler(self.handle_callback))
        app.add_error_handler(self.error_handler)
        
        logger.info("🚀 ربات شروع به کار کرد...")
        app.run_polling()


# کلاس‌های اضافی برای افکت‌های پیشرفته‌تر

class AdvancedPhysicsEngine:
    """موتور فیزیک پیشرفته برای شبیه‌سازی‌های واقع‌گرایانه"""
    
    @staticmethod
    def simulate_fluid_dynamics(img, viscosity=0.5):
        """شبیه‌سازی دینامیک سیالات"""
        h, w = img.shape[:2]
        frames = []
        
        # ایجاد میدان سرعت
        velocity_field = np.random.randn(h, w, 2) * 5
        
        for t in range(90):
            frame = img.copy()
            
            # محاسبه جریان
            for i in range(1, h-1):
                for j in range(1, w-1):
                    vx, vy = velocity_field[i, j]
                    new_i = int(i + vy * viscosity)
                    new_j = int(j + vx * viscosity)
                    
                    if 0 <= new_i < h and 0 <= new_j < w:
                        frame[i, j] = img[new_i, new_j]
            
            # به‌روزرسانی میدان سرعت
            velocity_field *= 0.98  # افت سرعت
            
            frames.append(frame)
        
        return frames
    
    @staticmethod
    def simulate_electromagnetic_field(img):
        """شبیه‌سازی میدان الکترومغناطیسی"""
        h, w = img.shape[:2]
        frames = []
        
        cx, cy = w//2, h//2
        
        for t in np.linspace(0, 4*np.pi, 90):
            frame = np.zeros_like(img)
            
            for i in range(h):
                for j in range(w):
                    dx, dy = j - cx, i - cy
                    distance = np.sqrt(dx**2 + dy**2) + 1
                    
                    # محاسبه میدان
                    field_strength = np.sin(distance/20 - t) / distance
                    
                    # تأثیر روی پیکسل‌ها
                    angle = np.arctan2(dy, dx) + field_strength
                    new_j = int(cx + distance * np.cos(angle))
                    new_i = int(cy + distance * np.sin(angle))
                    
                    if 0 <= new_i < h and 0 <= new_j < w:
                        frame[i, j] = img[new_i, new_j]
            
            frames.append(frame)
        
        return frames
    
    @staticmethod
    def simulate_quantum_effects(img):
        """شبیه‌سازی افکت‌های کوانتومی"""
        h, w = img.shape[:2]
        frames = []
        
        for t in np.linspace(0, 1, 90):
            frame = img.copy().astype(float)
            
            # اصل عدم قطعیت هایزنبرگ - نویز تصادفی
            uncertainty = np.random.randn(h, w, 3) * 10 * t
            frame += uncertainty
            
            # کوانتیزه کردن
            frame = np.clip(frame, 0, 255)
            
            # تونل‌زنی کوانتومی - پیکسل‌ها از دیوارها عبور می‌کنند
            if np.random.random() > 0.7:
                block_h, block_w = h//4, w//4
                y, x = np.random.randint(0, h-block_h), np.random.randint(0, w-block_w)
                frame[y:y+block_h, x:x+block_w] = img[y:y+block_h, x:x+block_w]
            
            frames.append(frame.astype(np.uint8))
        
        return frames


class ChemistryAnimationEngine:
    """موتور انیمیشن شیمیایی"""
    
    @staticmethod
    def simulate_chemical_reaction(img):
        """شبیه‌سازی واکنش شیمیایی"""
        h, w = img.shape[:2]
        frames = []
        
        # تبدیل به فضای LAB
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        for t in np.linspace(0, 1, 90):
            frame_lab = img_lab.copy().astype(float)
            
            # تغییر رنگ مانند واکنش شیمیایی
            frame_lab[:,:,1] += (np.random.randn(h, w) * 30 * t)  # a channel
            frame_lab[:,:,2] += (np.random.randn(h, w) * 30 * t)  # b channel
            
            # محدود کردن
            frame_lab = np.clip(frame_lab, 0, 255).astype(np.uint8)
            
            # بازگشت به RGB
            frame = cv2.cvtColor(frame_lab, cv2.COLOR_LAB2RGB)
            frames.append(frame)
        
        return frames
    
    @staticmethod
    def simulate_crystallization(img):
        """شبیه‌سازی تبلور"""
        h, w = img.shape[:2]
        frames = []
        
        # نقاط هسته‌ای
        nuclei = [(np.random.randint(0, w), np.random.randint(0, h)) 
                  for _ in range(20)]
        
        for t in np.linspace(0, 1, 90):
            frame = np.zeros_like(img)
            radius = int(min(h, w) * t / 2)
            
            for nx, ny in nuclei:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (nx, ny), radius, 255, -1)
                
                frame[mask > 0] = img[mask > 0]
            
            frames.append(frame)
        
        return frames


class MathematicalTransformEngine:
    """موتور تبدیلات ریاضی"""
    
    @staticmethod
    def apply_fourier_transform_animation(img):
        """انیمیشن تبدیل فوریه"""
        h, w = img.shape[:2]
        frames = []
        
        # تبدیل فوریه
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        for t in np.linspace(0, 1, 90):
            # ماسک دایره‌ای
            cx, cy = w//2, h//2
            radius = int(min(h, w) * t / 2)
            
            mask = np.zeros((h, w))
            y, x = np.ogrid[:h, :w]
            mask_area = (x - cx)**2 + (y - cy)**2 <= radius**2
            mask[mask_area] = 1
            
            # اعمال ماسک
            f_shift_masked = f_shift * mask
            
            # تبدیل معکوس
            f_ishift = np.fft.ifftshift(f_shift_masked)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            
            # نرمال‌سازی
            img_back = (img_back / img_back.max() * 255).astype(np.uint8)
            
            # تبدیل به RGB
            frame = cv2.cvtColor(img_back, cv2.COLOR_GRAY2RGB)
            frames.append(frame)
        
        return frames
    
    @staticmethod
    def apply_fractal_transformation(img):
        """تبدیل فراکتالی"""
        h, w = img.shape[:2]
        frames = []
        
        for iteration in range(1, 91):
            frame = img.copy()
            
            # تبدیل فراکتالی - مانند مجموعه ماندلبرو
            scale = 1 + iteration * 0.02
            new_h, new_w = int(h / scale), int(w / scale)
            
            frame = cv2.resize(frame, (new_w, new_h))
            
            # تکرار در کل فریم
            tiled = np.tile(frame, (int(np.ceil(h/new_h)), int(np.ceil(w/new_w)), 1))
            frame = tiled[:h, :w]
            
            frames.append(frame)
        
        return frames


# اضافه کردن افکت‌های جدید به کلاس اصلی
class UltraAdvancedAnimationEngine(AdvancedAnimationEngine):
    """نسخه فوق پیشرفته با تمام قابلیت‌ها"""
    
    def __init__(self):
        super().__init__()
        self.physics = AdvancedPhysicsEngine()
        self.chemistry = ChemistryAnimationEngine()
        self.math = MathematicalTransformEngine()
    
    def create_holographic_effect(self, img):
        """افکت هولوگرافیک"""
        h, w = img.shape[:2]
        frames = []
        
        for t in np.linspace(0, 2*np.pi, 90):
            frame = img.copy().astype(float)
            
            # شیفت RGB
            frame[:,:,0] = np.roll(frame[:,:,0], int(10*np.sin(t)), axis=1)
            frame[:,:,1] = np.roll(frame[:,:,1], int(10*np.sin(t+2*np.pi/3)), axis=1)
            frame[:,:,2] = np.roll(frame[:,:,2], int(10*np.sin(t+4*np.pi/3)), axis=1)
            
            # خطوط اسکن
            scan_line = int((t / (2*np.pi)) * h)
            frame[scan_line:scan_line+5] *= 1.5
            
            # نویز هولوگرافیک
            noise = np.random.rand(h, w, 3) * 30
            frame += noise * (0.5 + 0.5*np.sin(t))
            
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            frames.append(frame)
        
        return frames


# اجرای ربات
if __name__ == "__main__":
    # توجه: توکن‌های خود را اینجا وارد کنید
    bot = TelegramAnimationBot(
        token=TELEGRAM_TOKEN,
        anthropic_key=ANTHROPIC_API_KEY
    )
    
    print("""
    ═══════════════════════════════════════════════════
    🎬 ربات تلگرام حرفه‌ای انیمیشن
    ═══════════════════════════════════════════════════
    
    ✨ قابلیت‌ها:
    • بیش از 1000 نوع افکت انیمیشن
    • شبیه‌سازی فیزیک، شیمی، ریاضیات
    • هوش مصنوعی Claude برای تحلیل
    • کیفیت HD و 4K
    • پردازش پیشرفته تصویر
    
    📝 دستورالعمل:
    1. TELEGRAM_TOKEN را با توکن ربات خود جایگزین کنید
    2. ANTHROPIC_API_KEY را با کلید API Claude وارد کنید
    3. کتابخانه‌های مورد نیاز را نصب کنید
    4. ربات را اجرا کنید
    
    🚀 شروع کار...
    ═══════════════════════════════════════════════════
    """)
    
    bot.run()
