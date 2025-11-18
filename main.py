import os
import yfinance as yf
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from groq import Groq
import sqlite3
from datetime import datetime

# Environment variables
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize clients
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
else:
    client = None

# Set matplotlib to use Agg backend (non-interactive)
plt.switch_backend('Agg')

# ======================= DATABASE SQLITE =======================

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    
    # Create table portfolio
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            kode TEXT NOT NULL,
            jumlah INTEGER NOT NULL,
            harga_beli REAL NOT NULL,
            tanggal_beli TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create table watchlist
    c.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            kode TEXT NOT NULL,
            tanggal_tambah TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, kode)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database SQLite initialized!")

# ======================= WATCHLIST FUNCTIONS =======================

def tambah_watchlist_db(user_id, kode):
    """Menambah saham ke watchlist"""
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    
    try:
        c.execute('''
            INSERT INTO watchlist (user_id, kode)
            VALUES (?, ?)
        ''', (user_id, kode))
        
        conn.commit()
        result = f"✅ {kode} ditambahkan ke watchlist"
    except sqlite3.IntegrityError:
        result = f"❌ {kode} sudah ada di watchlist"
    except Exception as e:
        result = f"❌ Error: {str(e)}"
    finally:
        conn.close()
    
    return result

def hapus_watchlist_db(user_id, kode):
    """Menghapus saham dari watchlist"""
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    
    c.execute('''
        DELETE FROM watchlist 
        WHERE user_id = ? AND kode = ?
    ''', (user_id, kode))
    
    affected = c.rowcount
    conn.commit()
    conn.close()
    
    if affected > 0:
        return f"✅ {kode} dihapus dari watchlist"
    else:
        return f"❌ {kode} tidak ditemukan di watchlist"

def get_watchlist_db(user_id):
    """Mendapatkan watchlist dari database"""
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    
    c.execute('''
        SELECT kode, tanggal_tambah 
        FROM watchlist 
        WHERE user_id = ?
        ORDER BY tanggal_tambah DESC
    ''', (user_id,))
    
    results = c.fetchall()
    conn.close()
    
    return [row[0] for row in results]

def get_watchlist_summary_db(user_id):
    """Mendapatkan ringkasan watchlist dengan data harga"""
    watchlist = get_watchlist_db(user_id)
    
    if not watchlist:
        return None
    
    watchlist_data = []
    
    for kode in watchlist:
        try:
            ticker = yf.Ticker(kode + ".JK")
            data = ticker.history(period="1d")
            
            if data.empty:
                watchlist_data.append({
                    'kode': kode,
                    'harga': 0,
                    'perubahan': 0,
                    'error': True
                })
                continue
            
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
            perubahan = ((current_price - prev_price) / prev_price) * 100
            
            watchlist_data.append({
                'kode': kode,
                'harga': current_price,
                'perubahan': perubahan,
                'error': False
            })
            
        except Exception as e:
            watchlist_data.append({
                'kode': kode,
                'harga': 0,
                'perubahan': 0,
                'error': True
            })
            continue
    
    return watchlist_data

# ======================= PORTFOLIO FUNCTIONS =======================

def tambah_portfolio_db(user_id, kode, jumlah, harga_beli):
    """Menambah saham ke portfolio database"""
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    
    # Cek apakah saham sudah ada
    c.execute('''
        SELECT * FROM portfolio 
        WHERE user_id = ? AND kode = ?
    ''', (user_id, kode))
    
    existing = c.fetchone()
    
    if existing:
        # Update existing stock - average price calculation
        total_jumlah = existing[3] + jumlah
        total_invest = (existing[3] * existing[4]) + (jumlah * harga_beli)
        avg_harga = total_invest / total_jumlah
        
        c.execute('''
            UPDATE portfolio 
            SET jumlah = ?, harga_beli = ?
            WHERE user_id = ? AND kode = ?
        ''', (total_jumlah, avg_harga, user_id, kode))
        
        result = f"✅ {kode} diperbarui: {total_jumlah} lot @Rp {avg_harga:,.0f}"
    else:
        # Tambah saham baru
        c.execute('''
            INSERT INTO portfolio (user_id, kode, jumlah, harga_beli)
            VALUES (?, ?, ?, ?)
        ''', (user_id, kode, jumlah, harga_beli))
        
        result = f"✅ {kode} ditambahkan: {jumlah} lot @Rp {harga_beli:,.0f}"
    
    conn.commit()
    conn.close()
    return result

def hapus_portfolio_db(user_id, kode):
    """Menghapus saham dari portfolio database"""
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    
    c.execute('''
        DELETE FROM portfolio 
        WHERE user_id = ? AND kode = ?
    ''', (user_id, kode))
    
    affected = c.rowcount
    conn.commit()
    conn.close()
    
    if affected > 0:
        return f"✅ {kode} dihapus dari portfolio"
    else:
        return f"❌ {kode} tidak ditemukan di portfolio"

def get_portfolio_db(user_id):
    """Mendapatkan portfolio dari database"""
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    
    c.execute('''
        SELECT kode, jumlah, harga_beli, tanggal_beli 
        FROM portfolio 
        WHERE user_id = ?
    ''', (user_id,))
    
    results = c.fetchall()
    conn.close()
    
    portfolio = []
    for row in results:
        portfolio.append({
            'kode': row[0],
            'jumlah': row[1],
            'harga_beli': row[2],
            'tanggal_beli': row[3]
        })
    
    return portfolio

def get_portfolio_summary_db(user_id):
    """Mendapatkan ringkasan portfolio dari database"""
    portfolio = get_portfolio_db(user_id)
    
    if not portfolio:
        return None
    
    total_investasi = 0
    total_sekarang = 0
    performance_data = []
    
    for item in portfolio:
        try:
            # Get current price
            ticker = yf.Ticker(item['kode'] + ".JK")
            data = ticker.history(period="1d")
            if data.empty:
                current_price = item['harga_beli']
            else:
                current_price = data['Close'].iloc[-1]
            
            # Calculate values
            investasi = item['jumlah'] * item['harga_beli'] * 100  # 1 lot = 100 shares
            nilai_sekarang = item['jumlah'] * current_price * 100
            profit_loss = nilai_sekarang - investasi
            profit_persen = (profit_loss / investasi) * 100 if investasi > 0 else 0
            
            total_investasi += investasi
            total_sekarang += nilai_sekarang
            
            performance_data.append({
                'kode': item['kode'],
                'jumlah': item['jumlah'],
                'harga_beli': item['harga_beli'],
                'harga_sekarang': current_price,
                'investasi': investasi,
                'nilai_sekarang': nilai_sekarang,
                'profit_loss': profit_loss,
                'profit_persen': profit_persen
            })
        except Exception as e:
            print(f"Error processing {item['kode']}: {e}")
            continue
    
    if not performance_data:
        return None
    
    total_profit = total_sekarang - total_investasi
    total_profit_persen = (total_profit / total_investasi) * 100 if total_investasi > 0 else 0
    
    return {
        'performance_data': performance_data,
        'total_investasi': total_investasi,
        'total_sekarang': total_sekarang,
        'total_profit': total_profit,
        'total_profit_persen': total_profit_persen
    }

# ======================= WATCHLIST COMMANDS =======================

async def watchlist_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lihat Watchlist"""
    user_id = update.effective_user.id
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        watchlist_summary = get_watchlist_summary_db(user_id)
        
        if not watchlist_summary:
            await update.message.reply_text("""
📋 *WATCHLIST KOSONG*

Anda belum memiliki saham di watchlist.

*💡 Cara tambah watchlist:*
`/addwatch BBCA` - Tambah BBCA ke watchlist
`/addwatch BBRI` - Tambah BBRI ke watchlist

*💡 Fitur Watchlist:*
• Pantau harga saham favorit
• Update real-time
• Cek kapan saja dengan `/watchlist`

*Contoh:*
`/addwatch BBCA` → Tambah BBCA
`/addwatch BBRI` → Tambah BBRI
`/addwatch BMRI` → Tambah BMRI
""", parse_mode="Markdown")
            return
        
        response = "📋 *MY WATCHLIST*\n\n"
        
        for stock in watchlist_summary:
            if stock['error']:
                response += f"❌ *{stock['kode']}* - Error fetch data\n"
            else:
                change_icon = "🟢" if stock['perubahan'] >= 0 else "🔴"
                response += f"{change_icon} *{stock['kode']}* - Rp {stock['harga']:,.0f} ({change_icon} {stock['perubahan']:+.2f}%)\n"
        
        response += f"\n📊 Total: {len(watchlist_summary)} saham"
        response += "\n\n💡 Gunakan `/analisa [KODE]` untuk analisis detail"
        response += "\n💾 Data tersimpan permanen di SQLite"
        
        await update.message.reply_text(response, parse_mode="Markdown")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def addwatch_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Tambah saham ke watchlist"""
    if len(context.args) != 1:
        await update.message.reply_text("""
📋 *TAMBAH SAHAM KE WATCHLIST*

Format: `/addwatch [KODE]`

Contoh:
`/addwatch BBCA` → Tambah BBCA ke watchlist
`/addwatch BBRI` → Tambah BBRI ke watchlist
`/addwatch BMRI` → Tambah BMRI ke watchlist

*Note:*
- Maksimal 20 saham per user
- Data tersimpan PERMANEN di database
- Cek dengan `/watchlist`
""", parse_mode="Markdown")
        return
    
    kode = context.args[0].upper()
    user_id = update.effective_user.id
    
    # Cek batas maksimal watchlist
    current_watchlist = get_watchlist_db(user_id)
    if len(current_watchlist) >= 20:
        await update.message.reply_text("❌ Watchlist sudah penuh (maksimal 20 saham). Hapus beberapa dengan `/delwatch [KODE]`")
        return
    
    # Validasi kode saham
    try:
        ticker = yf.Ticker(kode + ".JK")
        data = ticker.history(period="1d")
        if data.empty:
            await update.message.reply_text(f"❌ Kode saham {kode} tidak valid atau tidak ditemukan.")
            return
    except:
        await update.message.reply_text(f"❌ Kode saham {kode} tidak valid.")
        return
    
    result = tambah_watchlist_db(user_id, kode)
    await update.message.reply_text(result)

async def delwatch_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Hapus saham dari watchlist"""
    if len(context.args) != 1:
        await update.message.reply_text("""
📋 *HAPUS SAHAM DARI WATCHLIST*

Format: `/delwatch [KODE]`

Contoh:
`/delwatch BBCA` → Hapus BBCA dari watchlist
`/delwatch BBRI` → Hapus BBRI dari watchlist

*Note:* Data akan dihapus PERMANEN dari database
""", parse_mode="Markdown")
        return
    
    kode = context.args[0].upper()
    user_id = update.effective_user.id
    
    result = hapus_watchlist_db(user_id, kode)
    await update.message.reply_text(result)

# ======================= INDICATOR FUNCTIONS =======================

def hitung_rsi(close, period=14):
    """Menghitung RSI dengan handling data insufficient"""
    if len(close) <= period:
        return None
    
    close = np.array(close)
    delta = np.diff(close)
    
    if len(delta) < period:
        return None
        
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.mean(gain[-period:])
    avg_loss = np.mean(loss[-period:])
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)

def hitung_ma(close, period=20):
    """Menghitung Moving Average"""
    if len(close) < period:
        return None
    return round(np.mean(close[-period:]), 2)

def hitung_macd(close):
    """Menghitung MACD sederhana"""
    if len(close) < 26:
        return None
    
    def ema(data, period):
        if len(data) < period:
            return None
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        return np.convolve(data, weights, mode='valid')[-1]
    
    ema_12 = ema(close, 12)
    ema_26 = ema(close, 26)
    
    if ema_12 is None or ema_26 is None:
        return None
        
    return round(ema_12 - ema_26, 3)

def hitung_volume_avg(volume, period=20):
    """Menghitung volume rata-rata"""
    if len(volume) < period:
        return None
    return np.mean(volume[-period:])

# ======================= PATTERN RECOGNITION =======================

def detect_pattern(data):
    """Mendeteksi pola candlestick"""
    if len(data) < 5:
        return "Data tidak cukup untuk analisis pattern"
    
    close = data['Close'].values
    open_ = data['Open'].values
    high = data['High'].values
    low = data['Low'].values
    volume = data['Volume'].values
    
    patterns = []
    
    # Bullish Engulfing
    if len(data) >= 2:
        prev_open, prev_close = open_[-2], close[-2]
        curr_open, curr_close = open_[-1], close[-1]
        
        if (prev_close < prev_open and  # Previous candle red
            curr_close > curr_open and  # Current candle green
            curr_open < prev_close and   # Open below prev close
            curr_close > prev_open):     # Close above prev open
            patterns.append("🟢 BULLISH ENGULFING")
    
    # Hammer
    if len(data) >= 1:
        body = abs(close[-1] - open_[-1])
        lower_wick = min(open_[-1], close[-1]) - low[-1]
        upper_wick = high[-1] - max(open_[-1], close[-1])
        
        if (lower_wick >= 2 * body and  # Long lower wick
            upper_wick <= body * 0.5 and  # Small upper wick
            close[-1] > open_[-1]):  # Green candle
            patterns.append("🔨 HAMMER (Bullish Reversal)")
    
    # Doji
    if len(data) >= 1:
        body = abs(close[-1] - open_[-1])
        high_low_range = high[-1] - low[-1]
        
        if body <= high_low_range * 0.1:  # Very small body
            patterns.append("🎯 DOJI (Indecision)")
    
    # Volume Analysis
    avg_volume = hitung_volume_avg(volume)
    if avg_volume and volume[-1] > avg_volume * 1.5:
        patterns.append("📈 VOLUME SPIKE")
    elif avg_volume and volume[-1] < avg_volume * 0.7:
        patterns.append("📉 VOLUME DRY UP")
    
    # Support Resistance
    support = np.min(low[-10:])
    resistance = np.max(high[-10:])
    
    if not patterns:
        patterns.append("🟡 NO CLEAR PATTERN")
    
    return {
        'patterns': patterns,
        'support': round(support, 2),
        'resistance': round(resistance, 2),
        'current_price': close[-1]
    }

# ======================= CUSTOM SCREENER =======================

def custom_screener(condition):
    """Screener custom berdasarkan kondisi"""
    tickers = get_idx_tickers()
    results = []
    
    for kode in tickers:
        try:
            data = yf.Ticker(kode).history(period="3mo")
            if data.empty or len(data) < 20:
                continue
                
            close = data['Close'].values
            volume = data['Volume'].values
            current_price = close[-1]
            
            # Calculate indicators
            rsi = hitung_rsi(close.tolist())
            ma20 = hitung_ma(close.tolist())
            ma50 = hitung_ma(close.tolist(), 50)
            macd = hitung_macd(close.tolist())
            avg_volume = hitung_volume_avg(volume)
            
            # Parse condition
            if eval_condition(condition, {
                'rsi': rsi, 'ma20': ma20, 'ma50': ma50, 
                'macd': macd, 'price': current_price,
                'volume': volume[-1], 'avg_volume': avg_volume
            }):
                results.append({
                    'kode': kode.replace('.JK', ''),
                    'price': current_price,
                    'rsi': rsi,
                    'ma20': ma20,
                    'volume_ratio': volume[-1] / avg_volume if avg_volume else 0
                })
                
        except Exception as e:
            continue
    
    return sorted(results, key=lambda x: x['rsi'] if x['rsi'] else 100)

def eval_condition(condition, variables):
    """Evaluate custom condition"""
    try:
        # Simple condition parser
        condition = condition.lower()
        
        # Replace variable names
        condition = condition.replace('rsi', str(variables['rsi'] or 0))
        condition = condition.replace('ma20', str(variables['ma20'] or 0))
        condition = condition.replace('ma50', str(variables['ma50'] or 0))
        condition = condition.replace('macd', str(variables['macd'] or 0))
        condition = condition.replace('price', str(variables['price'] or 0))
        condition = condition.replace('volume', str(variables['volume'] or 0))
        condition = condition.replace('avg_volume', str(variables['avg_volume'] or 1))
        
        # Evaluate safely
        return eval(condition)
    except:
        return False

# ======================= TRADING PLAN GENERATOR =======================

def generate_trading_plan(kode, data):
    """Generate trading plan otomatis"""
    if data.empty or len(data) < 20:
        return "Data tidak cukup untuk generate trading plan"
    
    close = data['Close'].values
    current_price = close[-1]
    
    # Calculate key levels
    support = np.min(close[-10:])
    resistance = np.max(close[-10:])
    
    # Risk management calculation
    risk_per_trade = 2  # 2% risk per trade
    stop_loss_percent = 3  # 3% stop loss
    
    stop_loss = current_price * (1 - stop_loss_percent / 100)
    take_profit1 = current_price * (1 + (stop_loss_percent * 1.5) / 100)
    take_profit2 = current_price * (1 + (stop_loss_percent * 2.5) / 100)
    
    risk_reward_ratio = (take_profit1 - current_price) / (current_price - stop_loss)
    
    return {
        'kode': kode,
        'current_price': round(current_price, 2),
        'entry_zone': f"{round(current_price * 0.99, 2)} - {round(current_price * 1.01, 2)}",
        'stop_loss': round(stop_loss, 2),
        'take_profit1': round(take_profit1, 2),
        'take_profit2': round(take_profit2, 2),
        'risk_reward_ratio': round(risk_reward_ratio, 2),
        'support': round(support, 2),
        'resistance': round(resistance, 2),
        'position_sizing': "Max 20% portfolio per trade"
    }

# ======================= ANALYSIS FUNCTION =======================

def analisa_ai(prompt):
    """Fungsi analisis AI dengan error handling"""
    if not client:
        return "❌ Groq API tidak tersedia. Pastikan GROQ_API_KEY sudah di-set."
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system", 
                    "content": "Kamu adalah analis saham profesional Indonesia. Berikan analisis yang objektif dan informatif dengan bahasa yang mudah dipahami."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error AI: {str(e)}"

# ======================= CHART GENERATOR =======================

def generate_chart(kode, period="3mo", interval="1d"):
    """Generate candlestick chart dengan multi-timeframe"""
    try:
        ticker = yf.Ticker(kode)
        
        period_map = {
            "1day": "1d",
            "1week": "1wk", 
            "1month": "1mo",
            "3months": "3mo",
            "6months": "6mo",
            "1year": "1y"
        }
        
        yf_period = period_map.get(period, "3mo")
        
        data = ticker.history(period=yf_period, interval=interval)
        
        if data.empty or len(data) < 5:
            return None
        
        mc = mpf.make_marketcolors(
            up='#2E8B57',
            down='#DC143C',
            edge={'up': 'green', 'down': 'red'},
            wick={'up': 'green', 'down': 'red'},
            volume={'up': '#2E8B57', 'down': '#DC143C'}
        )
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle=":")
        
        title_map = {
            "1day": f"{kode} - 1 Day ({interval})",
            "1week": f"{kode} - 1 Week ({interval})",
            "1month": f"{kode} - 1 Month",
            "3months": f"{kode} - 3 Months",
            "6months": f"{kode} - 6 Months", 
            "1year": f"{kode} - 1 Year"
        }
        
        title = title_map.get(period, f"{kode} - Chart")
        
        use_ma = interval == "1d"
        
        mpf.plot(
            data,
            type='candle',
            mav=(20, 50) if use_ma else None,
            volume=True,
            style=s,
            title=title,
            ylabel='Price (Rp)',
            ylabel_lower='Volume',
            savefig='chart.png',
            figsize=(12, 8),
            tight_layout=True
        )
        return "chart.png"
    except Exception as e:
        print(f"Chart error: {e}")
        return None

# ======================= SCREENER FUNCTIONS =======================

def get_idx_tickers():
    """Get list of Indonesian stock tickers"""
    common_tickers = [
        'BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'BBNI.JK', 
        'TLKM.JK', 'ASII.JK', 'UNVR.JK', 'ICBP.JK',
        'INDF.JK', 'UNTR.JK', 'ADRO.JK', 'ANTM.JK',
        'PGAS.JK', 'PTBA.JK', 'AKRA.JK', 'MEDC.JK',
        'HRUM.JK', 'ITMG.JK', 'CPIN.JK', 'SMGR.JK'
    ]
    return common_tickers

def screener_oversold():
    """Screener untuk saham oversold (RSI < 30)"""
    hasil = []
    tickers = get_idx_tickers()
    
    for kode in tickers:
        try:
            data = yf.Ticker(kode).history(period="3mo")
            if data.empty or len(data) < 15:
                continue
                
            close_prices = data["Close"].tolist()
            rsi = hitung_rsi(close_prices)
            
            if rsi is not None and rsi < 30:
                current_price = close_prices[-1]
                price_change = ((close_prices[-1] - close_prices[-2]) / close_prices[-2]) * 100
                hasil.append((kode.replace(".JK", ""), rsi, current_price, round(price_change, 2)))
                
        except Exception as e:
            continue
    
    return sorted(hasil, key=lambda x: x[1])

# ======================= COMMAND: HELP & START =======================

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Menu bantuan"""
    menu = """
🤖 *Bot Mentor Saham IDX* 📈

*/start* - Memulai bot
*/help* - Menu bantuan

*/analisa [KODE]* - Analisa lengkap + entry/exit/SL
*/chart [KODE] [TIMEFRAME]* - Chart Candlestick
*/pattern [KODE]* - Pattern Recognition

*💼 PORTFOLIO:*
*/portfolio* - Lihat Portfolio
*/addportfolio [KODE] [LOT] [HARGA_BELI]* - Tambah Saham
*/delportfolio [KODE]* - Hapus Saham

*📋 WATCHLIST:*
*/watchlist* - Lihat Watchlist
*/addwatch [KODE]* - Tambah ke Watchlist  
*/delwatch [KODE]* - Hapus dari Watchlist

*🔍 SCREENER:*
*/screener [TYPE]* - Screener Saham
*/tradingplan [KODE]* - Trading Plan Generator

*📊 Timeframe Chart:*
• `/chart BBCA` - 3 bulan
• `/chart BBCA 1day` - 1 hari
• `/chart BBCA 1week` - 1 minggu  

*🔍 Screener Types:*
• `/screener oversold` - RSI < 30
• `/screener custom "rsi<30 and price>ma20"` - Custom

*💼 Portfolio Commands:*
• `/portfolio` - Lihat portfolio
• `/addportfolio BBCA 10 8500` - Beli 10 lot BBCA @8500
• `/delportfolio BBCA` - Jual/hapus BBCA

*📋 Watchlist Commands:*
• `/watchlist` - Lihat watchlist
• `/addwatch BBCA` - Tambah BBCA ke watchlist
• `/delwatch BBCA` - Hapus BBCA dari watchlist

*📚 Contoh:*
/analisa BBCA
/pattern BBRI
/addportfolio BBCA 5 8750
/portfolio
/addwatch BMRI
/watchlist
/screener custom "rsi<40 and price>ma20"
/tradingplan BMRI

*💾 FITUR BARU:* 
• Data portfolio TERSIMPAN PERMANEN di SQLite!
• WATCHLIST untuk pantau saham favorit!
"""
    await update.message.reply_text(menu, parse_mode="Markdown")

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Command start"""
    welcome = """
🎯 *Selamat Datang di Bot Mentor Saham!* 🤖

Sekarang dengan fitur **PORTFOLIO TRACKER + WATCHLIST + SQLITE**:

• 💾 *Data Permanen* - Portfolio & watchlist gak ilang kalo bot restart!
• 💼 *Portfolio Management* - Track profit/loss real-time
• 📋 *Watchlist* - Pantau saham favorit
• 🕯️ *Pattern Recognition* - Deteksi pola candlestick
• 🔍 *Custom Screener* - Cari saham sesuai kriteria
• 📝 *Trading Plan* - Rencana trading otomatis

*💡 Mulai dengan:*
/addwatch BBCA → Tambah ke watchlist
/watchlist → Lihat watchlist
/addportfolio BBCA 10 8500 → Tambah ke portfolio
/portfolio → Lihat kinerja
/analisa BBCA → Analisa lengkap

Gunakan /help untuk menu lengkap!
"""
    await update.message.reply_text(welcome, parse_mode="Markdown")

# ======================= COMMAND: ANALISA =======================

async def analisa_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Analisa saham lengkap"""
    if len(context.args) == 0:
        await update.message.reply_text("❌ Format: `/analisa BBCA`", parse_mode="Markdown")
        return
    
    kode = context.args[0].upper()
    full_kode = kode + ".JK"
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        ticker = yf.Ticker(full_kode)
        data = ticker.history(period="3mo")
        
        if data.empty:
            await update.message.reply_text("❌ Kode saham tidak ditemukan atau tidak ada data.")
            return
        
        close_prices = data["Close"].tolist()
        last_price = close_prices[-1]
        prev_price = close_prices[-2] if len(close_prices) > 1 else last_price
        price_change_pct = ((last_price - prev_price) / prev_price) * 100
        
        # Hitung indikator
        rsi = hitung_rsi(close_prices)
        ma20 = hitung_ma(close_prices, 20)
        ma50 = hitung_ma(close_prices, 50)
        macd = hitung_macd(close_prices)
        
        # Tentukan sinyal RSI
        rsi_signal = "🟢 OVERSOLD" if rsi and rsi < 30 else "🔴 OVERBOUGHT" if rsi and rsi > 70 else "🟡 NETRAL"
        
        # Tentukan sinyal MA
        ma_signal = "🟢 BULLISH" if ma20 and ma50 and ma20 > ma50 else "🔴 BEARISH" if ma20 and ma50 and ma20 < ma50 else "🟡 SIDEWAYS"
        
        # Format indikator
        indikator = f"""
📊 *ANALISIS TEKNIKAL {kode}*

💵 *Harga:* Rp {last_price:,.0f}
📈 *Perubahan:* {price_change_pct:+.2f}%

🎯 *INDIKATOR:*
• RSI 14: *{rsi}* - {rsi_signal}
• MA 20: Rp {ma20:,.0f} 
• MA 50: Rp {ma50:,.0f}
• MACD: *{macd}*
• Sinyal MA: {ma_signal}
"""
        # AI Analysis
        prompt = f"""
Analisa saham {kode} dengan data teknikal berikut:

DATA TEKNIKAL:
- Harga terakhir: Rp {last_price:,.0f}
- Perubahan harga: {price_change_pct:+.2f}%
- RSI 14: {rsi} ({rsi_signal})
- MA 20: Rp {ma20:,.0f}
- MA 50: Rp {ma50:,.0f} 
- MACD: {macd}
- Sinyal Moving Average: {ma_signal}

Berikan analisis teknikal komprehensif dengan format:

1. 📈 TREN SAAT INI: 
   [Jelaskan trend short-medium term]

2. 🎯 LEVEL KUNCI:
   Support: [level support]
   Resistance: [level resistance]

3. 💰 REKOMENDASI TRADING:
   Entry: [harga entry spesifik]
   Target Profit: [harga TP spesifik] 
   Stop Loss: [harga SL spesifik]

4. ⚠️ MANAJEMEN RISIKO:
   Level Risiko: [rendah/sedang/tinggi]
   Potensi Reward/Risk: [ratio R/R]

5. ✅ KESIMPULAN:
   [Ringkasan singkat dan clear]

Gunakan bahasa Indonesia yang mudah dipahami dan berikan angka spesifik!
"""
        ai_analysis = analisa_ai(prompt)
        
        response = indikator + "\n" + "🤖" * 5 + "\n*ANALISIS MENTOR AI:*\n" + ai_analysis
        await update.message.reply_text(response, parse_mode="Markdown")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

# ======================= COMMAND: PORTFOLIO TRACKER =======================

async def portfolio_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lihat Portfolio"""
    user_id = update.effective_user.id
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        portfolio_summary = get_portfolio_summary_db(user_id)
        
        if not portfolio_summary:
            await update.message.reply_text("""
💼 *PORTFOLIO KOSONG*

Anda belum memiliki saham di portfolio.

*💡 Cara tambah saham:*
`/addportfolio BBCA 10 8500`
- BBCA = Kode saham
- 10 = Jumlah lot  
- 8500 = Harga beli per saham

*Contoh:*
`/addportfolio BBRI 5 4200` → Beli 5 lot BBRI @4200
`/addportfolio BMRI 8 5200` → Beli 8 lot BMRI @5200

*💾 Data tersimpan permanen di SQLite!*
""", parse_mode="Markdown")
            return
        
        response = f"""
💼 *PORTFOLIO SUMMARY*

"""
        # Add each stock performance
        for stock in portfolio_summary['performance_data']:
            profit_icon = "🟢" if stock['profit_loss'] >= 0 else "🔴"
            response += f"""
{profit_icon} *{stock['kode']}*
{stock['jumlah']} lot • Beli: Rp {stock['harga_beli']:,.0f} • Sekarang: Rp {stock['harga_sekarang']:,.0f}
Investasi: Rp {stock['investasi']:,.0f} • Nilai: Rp {stock['nilai_sekarang']:,.0f}
P/L: {profit_icon} Rp {stock['profit_loss']:,.0f} ({stock['profit_persen']:+.1f}%)

"""
        
        # Add total summary
        total_icon = "🟢" if portfolio_summary['total_profit'] >= 0 else "🔴"
        response += f"""
📊 *TOTAL PORTFOLIO:*
Total Investasi: Rp {portfolio_summary['total_investasi']:,.0f}
Nilai Sekarang: Rp {portfolio_summary['total_sekarang']:,.0f}
Total Profit/Loss: {total_icon} Rp {portfolio_summary['total_profit']:,.0f} ({portfolio_summary['total_profit_persen']:+.1f}%)

💾 Data tersimpan permanen di SQLite
💡 Gunakan `/analisa [KODE]` untuk analisis lanjutan
"""
        await update.message.reply_text(response, parse_mode="Markdown")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def addportfolio_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Tambah saham ke portfolio"""
    if len(context.args) != 3:
        await update.message.reply_text("""
💼 *TAMBAH SAHAM KE PORTFOLIO*

Format: `/addportfolio [KODE] [LOT] [HARGA_BELI]`

Contoh:
`/addportfolio BBCA 10 8500` → Beli 10 lot BBCA @8500
`/addportfolio BBRI 5 4200` → Beli 5 lot BBRI @4200
`/addportfolio BMRI 8 5200` → Beli 8 lot BMRI @5200

*Keterangan:*
- 1 lot = 100 lembar saham
- Harga beli per lembar saham
- 💾 Data tersimpan PERMANEN di SQLite
""", parse_mode="Markdown")
        return
    
    try:
        kode = context.args[0].upper()
        jumlah = int(context.args[1])
        harga_beli = float(context.args[2])
        user_id = update.effective_user.id
        
        if jumlah <= 0 or harga_beli <= 0:
            await update.message.reply_text("❌ Jumlah dan harga harus positif")
            return
        
        result = tambah_portfolio_db(user_id, kode, jumlah, harga_beli)
        await update.message.reply_text(result + "\n\n💾 Data tersimpan di SQLite!")
        
    except ValueError:
        await update.message.reply_text("❌ Format angka tidak valid. Gunakan: /addportfolio BBCA 10 8500")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def delportfolio_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Hapus saham dari portfolio"""
    if len(context.args) != 1:
        await update.message.reply_text("""
💼 *HAPUS SAHAM DARI PORTFOLIO*

Format: `/delportfolio [KODE]`

Contoh:
`/delportfolio BBCA` → Jual/hapus BBCA dari portfolio
`/delportfolio BBRI` → Jual/hapus BBRI dari portfolio

*Note:* Data akan dihapus PERMANEN dari database
""", parse_mode="Markdown")
        return
    
    kode = context.args[0].upper()
    user_id = update.effective_user.id
    
    result = hapus_portfolio_db(user_id, kode)
    await update.message.reply_text(result)

# ======================= COMMAND: PATTERN RECOGNITION =======================

async def pattern_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Pattern Recognition"""
    if len(context.args) == 0:
        await update.message.reply_text("❌ Format: `/pattern BBCA`", parse_mode="Markdown")
        return
    
    kode = context.args[0].upper()
    full_kode = kode + ".JK"
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        ticker = yf.Ticker(full_kode)
        data = ticker.history(period="1mo")
        
        if data.empty:
            await update.message.reply_text("❌ Kode saham tidak ditemukan.")
            return
        
        pattern_result = detect_pattern(data)
        
        response = f"""
🕯️ *PATTERN RECOGNITION - {kode}*

💵 *Harga Saat Ini:* Rp {pattern_result['current_price']:,.0f}

*🎯 POLA TERDETEKSI:*
"""
        for pattern in pattern_result['patterns']:
            response += f"• {pattern}\n"
        
        response += f"""
*📊 LEVEL KUNCI:*
• Support: Rp {pattern_result['support']:,.0f}
• Resistance: Rp {pattern_result['resistance']:,.0f}

*💡 INTERPRETASI:*
"""
        if "BULLISH" in str(pattern_result['patterns']):
            response += "• Potensi reversal naik 🟢\n• Pertimbangkan entry di support"
        elif "HAMMER" in str(pattern_result['patterns']):
            response += "• Biasanya di akhir trend turun 🔨\n• Konfirmasi dengan volume"
        elif "DOJI" in str(pattern_result['patterns']):
            response += "• Market indecision 🎯\n• Tunggu konfirmasi breakout"
        else:
            response += "• Tidak ada sinyal kuat 🟡\n• Tunggu pattern lebih jelas"
        
        await update.message.reply_text(response, parse_mode="Markdown")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

# ======================= COMMAND: CUSTOM SCREENER =======================

async def screener_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Screener saham"""
    if len(context.args) == 0:
        await update.message.reply_text("""
🔍 *SCREENER SAHAM*

*Format:*
`/screener oversold` - Saham RSI < 30
`/screener custom "kondisi"` - Custom screener

*Contoh Custom:*
`/screener custom "rsi<30"` - RSI bawah 30
`/screener custom "price>ma20 and rsi<40"` - Price above MA20 & RSI<40
`/screener custom "volume>avg_volume*1.5"` - Volume spike

*Variabel yang tersedia:*
• rsi, ma20, ma50, macd, price, volume, avg_volume
""", parse_mode="Markdown")
        return
    
    mode = context.args[0].lower()
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    if mode == "oversold":
        hasil = screener_oversold()
        if not hasil:
            await update.message.reply_text("✅ Tidak ada saham oversold (RSI < 30) saat ini.")
            return
        
        teks = "📉 *SAHAM OVERSOLD (RSI < 30):*\n\n"
        for kode, rsi, price, change in hasil[:10]:
            change_icon = "🟢" if change > 0 else "🔴"
            teks += f"• *{kode}* - RSI: {rsi} - Harga: Rp {price:,.0f} ({change_icon} {change:+.1f}%)\n"
            
        teks += "\n💡 *Tips:* Saham oversold berpotensi rebound!"
        
        await update.message.reply_text(teks, parse_mode="Markdown")
        
    elif mode == "custom" and len(context.args) > 1:
        condition = " ".join(context.args[1:])
        hasil = custom_screener(condition)
        
        if not hasil:
            await update.message.reply_text("❌ Tidak ada saham yang memenuhi kriteria.")
            return
            
        teks = f"🔍 *CUSTOM SCREENER:* `{condition}`\n\n"
        for stock in hasil[:8]:
            volume_icon = "📈" if stock['volume_ratio'] > 1.2 else "📉"
            teks += f"• *{stock['kode']}* - RSI: {stock['rsi']} - Price: Rp {stock['price']:,.0f} {volume_icon}\n"
            
        teks += f"\n📊 Ditemukan {len(hasil)} saham"
        
        await update.message.reply_text(teks, parse_mode="Markdown")
        
    else:
        await update.message.reply_text("❌ Mode screener tidak dikenali.")

# ======================= COMMAND: TRADING PLAN =======================

async def tradingplan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Trading Plan Generator"""
    if len(context.args) == 0:
        await update.message.reply_text("❌ Format: `/tradingplan BBCA`", parse_mode="Markdown")
        return
    
    kode = context.args[0].upper()
    full_kode = kode + ".JK"
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        ticker = yf.Ticker(full_kode)
        data = ticker.history(period="1mo")
        
        if data.empty:
            await update.message.reply_text("❌ Kode saham tidak ditemukan.")
            return
        
        plan = generate_trading_plan(kode, data)
        
        if isinstance(plan, str):
            await update.message.reply_text(plan)
            return
        
        response = f"""
📝 *TRADING PLAN - {kode}*

💵 *Current Price:* Rp {plan['current_price']:,.0f}

*🎯 TRADING SETUP:*
• Entry Zone: Rp {plan['entry_zone']}
• Stop Loss: Rp {plan['stop_loss']:,.0f}
• Take Profit 1: Rp {plan['take_profit1']:,.0f}
• Take Profit 2: Rp {plan['take_profit2']:,.0f}

*📊 RISK MANAGEMENT:*
• Risk/Reward Ratio: {plan['risk_reward_ratio']}:1
• {plan['position_sizing']}
• Max Portfolio Risk: 2%

*🛡️ LEVEL KUNCI:*
• Support: Rp {plan['support']:,.0f}
• Resistance: Rp {plan['resistance']:,.0f}

*💡 EXECUTION PLAN:*
1. Entry di zone {plan['entry_zone']}
2. SL ketat di Rp {plan['stop_loss']:,.0f}
3. TP1: 50% position @Rp {plan['take_profit1']:,.0f}
4. TP2: 50% position @Rp {plan['take_profit2']:,.0f}

"""
        if plan['risk_reward_ratio'] >= 2:
            response += "✅ *SETUP BAGUS* - R/R Ratio optimal"
        elif plan['risk_reward_ratio'] >= 1:
            response += "🟡 *SETUP CUKUP* - Pertimbangkan carefully"
        else:
            response += "🔴 *SETUP BURUK* - Cari setup lain"
            
        await update.message.reply_text(response, parse_mode="Markdown")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

# ======================= COMMAND: CHART =======================

async def chart_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate chart candlestick dengan multi-timeframe"""
    if len(context.args) == 0:
        await update.message.reply_text("""
📊 *FORMAT CHART:*

`/chart BBCA` → Chart 3 bulan (default)
`/chart BBCA 1day` → Chart 1 hari
`/chart BBCA 1week` → Chart 1 minggu  
`/chart BBCA 1month` → Chart 1 bulan
`/chart BBCA 1year` → Chart 1 tahun

*Contoh:* `/chart BBRI 1week`
""", parse_mode="Markdown")
        return
    
    kode = context.args[0].upper()
    full_kode = kode + ".JK"
    
    period = "3months"
    if len(context.args) > 1:
        period_arg = context.args[1].lower()
        period_map = {
            "1day": "1day", "1d": "1day", "day": "1day",
            "1week": "1week", "1w": "1week", "week": "1week",
            "1month": "1month", "1m": "1month", "month": "1month",
            "3months": "3months", "3m": "3months", "3month": "3months",
            "1year": "1year", "1y": "1year", "year": "1year"
        }
        period = period_map.get(period_arg, "3months")
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_photo")
    
    try:
        file_path = generate_chart(full_kode, period)
        if file_path and os.path.exists(file_path):
            with open(file_path, 'rb') as chart_file:
                caption_map = {
                    "1day": f"📈 {kode} - Chart 1 Hari",
                    "1week": f"📈 {kode} - Chart 1 Minggu", 
                    "1month": f"📈 {kode} - Chart 1 Bulan",
                    "3months": f"📈 {kode} - Chart 3 Bulan",
                    "1year": f"📈 {kode} - Chart 1 Tahun"
                }
                caption = caption_map.get(period, f"📈 {kode} - Chart")
                
                await update.message.reply_photo(
                    photo=InputFile(chart_file),
                    caption=caption
                )
            os.remove(file_path)
        else:
            await update.message.reply_text("❌ Gagal membuat chart. Pastikan kode saham benar.")
    except Exception as e:
        await update.message.reply_text(f"❌ Error generating chart: {str(e)}")

# ======================= BOT SETUP =======================

def main():
    """Main function to run the bot"""
    if not BOT_TOKEN:
        print("❌ ERROR: BOT_TOKEN tidak ditemukan! Pastikan sudah di-set di Environment Variables.")
        return
    
    # Initialize database
    init_database()
    
    print("🤖 Initializing Bot Mentor Saham dengan SQLite Database + Watchlist...")
    
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("analisa", analisa_cmd))
    app.add_handler(CommandHandler("chart", chart_cmd))
    app.add_handler(CommandHandler("pattern", pattern_cmd))
    app.add_handler(CommandHandler("portfolio", portfolio_cmd))
    app.add_handler(CommandHandler("addportfolio", addportfolio_cmd))
    app.add_handler(CommandHandler("delportfolio", delportfolio_cmd))
    app.add_handler(CommandHandler("watchlist", watchlist_cmd))
    app.add_handler(CommandHandler("addwatch", addwatch_cmd))
    app.add_handler(CommandHandler("delwatch", delwatch_cmd))
    app.add_handler(CommandHandler("screener", screener_cmd))
    app.add_handler(CommandHandler("tradingplan", tradingplan_cmd))
    
    print("✅ Bot Mentor Saham dengan SQLITE DATABASE + WATCHLIST berhasil diinisialisasi!")
    print("💾 Data portfolio & watchlist tersimpan permanen di portfolio.db")
    print("🎯 Fitur: Portfolio Tracker, Watchlist, Pattern Recognition, Custom Screener, Trading Plan")
    print("🚀 Bot sedang berjalan...")
    print("💡 Gunakan /start di Telegram untuk mulai!")
    
    app.run_polling()

if __name__ == "__main__":
    main()