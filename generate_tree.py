import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Font configuration
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'

# Create figure (2000 x 1400 px at 150 DPI)
fig, ax = plt.subplots(figsize=(20, 14), dpi=150)
fig.patch.set_facecolor('#0B0F19')
ax.set_facecolor('#0B0F19')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Color palette (Modern Dark Theme)
BG_NODE = '#131C2E'
BORDER_NODE = '#2563EB'
TEXT_COLOR = '#F1F5F9'
TEXT_MUTED = '#94A3B8'

COLOR_GREEN_BG = '#064E3B'
COLOR_GREEN_BORDER = '#10B981'
COLOR_RED_BG = '#4C0519'
COLOR_RED_BORDER = '#F43F5E'
COLOR_AMBER_BG = '#451A03'
COLOR_AMBER_BORDER = '#F59E0B'
COLOR_GRAY_BG = '#1E293B'
COLOR_GRAY_BORDER = '#475569'
COLOR_BLUE_BG = '#1E3A8A'
COLOR_BLUE_BORDER = '#38BDF8'

def draw_card(ax, x, y, w, h, bg_color, border_color, title, subtitle="", align='center', title_size=11, sub_size=9.5, lw=1.5):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.3,rounding_size=1.0",
                       fc=bg_color, ec=border_color, lw=lw, zorder=2)
    ax.add_patch(p)
    
    cx = x + w / 2 if align == 'center' else x + 1.2
    
    if subtitle:
        cy_title = y + h * 0.65
        cy_sub = y + h * 0.32
        ax.text(cx, cy_title, title, ha=align, va='center',
                fontsize=title_size, fontweight='bold', color='#FFFFFF', zorder=3)
        ax.text(cx, cy_sub, subtitle, ha=align, va='center',
                fontsize=sub_size, color=TEXT_COLOR, zorder=3, linespacing=1.25)
    else:
        cy = y + h * 0.5
        ax.text(cx, cy, title, ha=align, va='center',
                fontsize=title_size, fontweight='bold', color='#FFFFFF', zorder=3, linespacing=1.25)

def draw_arrow(ax, x1, y1, x2, y2, color='#38BDF8', lw=2, label="", label_pos=0.5, label_bg=None, label_color='#FFFFFF', mut_scale=10):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='-|>,head_length=4,head_width=3',
                            mutation_scale=mut_scale, lw=lw, color=color, zorder=1)
    ax.add_patch(arrow)
    if label:
        lx = x1 + (x2 - x1) * label_pos
        ly = y1 + (y2 - y1) * label_pos
        bbox_props = dict(boxstyle="round,pad=0.25", fc=label_bg if label_bg else '#0B0F19', ec=color, lw=1)
        ax.text(lx, ly, label, ha='center', va='center', fontsize=9, fontweight='bold',
                color=label_color, bbox=bbox_props, zorder=4)

# ----------------- HEADER -----------------
ax.text(50, 97.2, "FLUSSO LOGICO DEL BOT — ALBERO DECISIONALE", 
        ha='center', va='center', fontsize=21, fontweight='heavy', color='#38BDF8')
ax.text(50, 95.0, "Strategia Trend-Following Donchian + Uscite Dinamiche su Candele a 4 Ore", 
        ha='center', va='center', fontsize=11.5, color='#94A3B8')

# ----------------- NODO INIZIALE -----------------
draw_card(ax, 32, 86.8, 36, 6.2, '#1E293B', '#38BDF8',
          "OGNI 4 ORE (Chiusura Candela)",
          "Il Bot controlla la moneta, aggiorna gli indicatori e decide l'azione",
          align='center', title_size=12, sub_size=10, lw=2)

draw_arrow(ax, 50, 86.8, 50, 82.8, color='#38BDF8', lw=2.2, mut_scale=12)

# ----------------- BIVIO PRINCIPALE -----------------
draw_card(ax, 34, 76.8, 32, 6.0, '#1E3A8A', '#60A5FA',
          "BIVIO: Ho già comprato questa moneta?",
          "Verifica se esiste una posizione aperta attiva nel portafoglio",
          align='center', title_size=11.5, sub_size=9.5, lw=2)

# Frecce Bivio verso le due colonne
draw_arrow(ax, 34, 79.8, 26, 73.8, color='#F59E0B', lw=2.2, label="  SÌ (In Posizione)  ", label_pos=0.45, label_bg='#78350F', label_color='#FDE68A', mut_scale=12)
draw_arrow(ax, 66, 79.8, 74, 73.8, color='#10B981', lw=2.2, label="  NO (Liquido in Euro)  ", label_pos=0.45, label_bg='#064E3B', label_color='#A7F3D0', mut_scale=12)

# =========================================================================
# RAMO A — SINISTRA (POSIZIONE APERTA: GESTIONE & USCITE)
# =========================================================================
draw_card(ax, 4, 68.2, 44, 4.8, '#1E293B', '#F59E0B',
          "RAMO A: SÌ (Posizione Aperta — Gestione Uscita e Stop Loss)",
          "Obiettivo: Proteggere il capitale e bloccare il profitto massimo",
          align='center', title_size=11, sub_size=9, lw=2)

# Step 1: Aggiornamento Trailing
draw_card(ax, 4, 59.8, 44, 6.2, '#0F172A', '#38BDF8',
          "1. Aggiornamento Massimi & Protezione Dinamica",
          "• Aggiorna il Prezzo Massimo raggiunto dal momento dell'acquisto\n• Alza la rete mobile di protezione: Chandelier = Massimo - 4.5× Volatilità (ATR)",
          align='left', title_size=10.5, sub_size=9, lw=1.5)

draw_arrow(ax, 26, 68.2, 26, 66.0, color='#94A3B8', lw=1.5, mut_scale=10)
draw_arrow(ax, 26, 59.8, 26, 56.5, color='#94A3B8', lw=1.5, mut_scale=10)

# Controllo 1: Stop Loss Fisso
draw_card(ax, 4, 49.8, 23, 6.2, '#1E293B', '#F43F5E',
          "Controllo 1: Stop Loss Fisso",
          "Il prezzo è crollato sotto\nlo Stop iniziale (-2.2× ATR)?",
          align='center', title_size=10, sub_size=8.5, lw=1.5)

draw_arrow(ax, 27, 52.9, 31, 52.9, color='#F43F5E', lw=2, label="SÌ", label_pos=0.5, label_bg='#4C0519', label_color='#FECDD3', mut_scale=8)

draw_card(ax, 31, 49.8, 17, 6.2, COLOR_RED_BG, COLOR_RED_BORDER,
          "VENDI SUBITO!",
          "Stop Loss Eseguito:\nPerdita bloccata al max 4%",
          align='center', title_size=10, sub_size=8.5, lw=2)

draw_arrow(ax, 15.5, 49.8, 15.5, 46.5, color='#94A3B8', lw=1.5, label="NO", label_pos=0.5, label_bg='#1E293B', label_color='#94A3B8', mut_scale=10)

# Controllo 2: Trailing Chandelier
draw_card(ax, 4, 39.8, 23, 6.2, '#1E293B', '#10B981',
          "Controllo 2: Rete Chandelier",
          "Il prezzo è sceso dal picco\ntoccando la rete mobile?",
          align='center', title_size=10, sub_size=8.5, lw=1.5)

draw_arrow(ax, 27, 42.9, 31, 42.9, color='#10B981', lw=2, label="SÌ", label_pos=0.5, label_bg='#064E3B', label_color='#A7F3D0', mut_scale=8)

draw_card(ax, 31, 39.8, 17, 6.2, COLOR_GREEN_BG, COLOR_GREEN_BORDER,
          "INCASSA IL PROFITTO!",
          "Vendi con Chandelier Stop:\nGuadagni protetti dal crollo",
          align='center', title_size=10, sub_size=8.5, lw=2)

draw_arrow(ax, 15.5, 39.8, 15.5, 36.5, color='#94A3B8', lw=1.5, label="NO", label_pos=0.5, label_bg='#1E293B', label_color='#94A3B8', mut_scale=10)

# Controllo 3: EMA 21 Esaurimento Trend
draw_card(ax, 4, 29.8, 23, 6.2, '#1E293B', '#F59E0B',
          "Controllo 3: Media EMA 21",
          "La candela ha chiuso sotto\nla media veloce EMA 21?",
          align='center', title_size=10, sub_size=8.5, lw=1.5)

draw_arrow(ax, 27, 32.9, 31, 32.9, color='#F59E0B', lw=2, label="SÌ", label_pos=0.5, label_bg='#451A03', label_color='#FDE68A', mut_scale=8)

draw_card(ax, 31, 29.8, 17, 6.2, COLOR_AMBER_BG, COLOR_AMBER_BORDER,
          "VENDI IN ANTICIPO!",
          "Il trend si è spento:\nUscita rapida di sicurezza",
          align='center', title_size=10, sub_size=8.5, lw=2)

draw_arrow(ax, 15.5, 29.8, 15.5, 25.5, color='#38BDF8', lw=1.8, label="TUTTI NO", label_pos=0.5, label_bg='#0F172A', label_color='#38BDF8', mut_scale=10)

# Esito A: Hold
draw_card(ax, 4, 17.5, 44, 7.5, '#0C4A6E', '#0284C7',
          "TIENI LA MONETA (HOLD) — NESSUN SEGNALE DI VENDITA",
          "• Il trend è intatto e la protezione mobile sale con il prezzo\n• Non fare nulla: lascia correre i profitti e aspetta le prossime 4 ore",
          align='center', title_size=11, sub_size=9.5, lw=2)


# =========================================================================
# RAMO B — DESTRA (LIQUIDITÀ IN EURO: RICERCA INGRESSI)
# =========================================================================
draw_card(ax, 52, 68.2, 44, 4.8, '#1E293B', '#10B981',
          "RAMO B: NO (Sono Liquido in Euro — Ricerca Ingressi)",
          "Obiettivo: Entrare solo su breakout confermati a forte probabilità",
          align='center', title_size=11, sub_size=9, lw=2)

# Circuit Breaker 24h
draw_card(ax, 52, 59.8, 23, 6.2, '#1E293B', '#F43F5E',
          "Controllo Sicurezza Globale",
          "Ho perso più dell'8% del conto\nnelle ultime 24 ore?",
          align='center', title_size=10, sub_size=8.5, lw=1.5)

draw_arrow(ax, 75, 62.9, 79, 62.9, color='#F43F5E', lw=2, label="SÌ", label_pos=0.5, label_bg='#4C0519', label_color='#FECDD3', mut_scale=8)

draw_card(ax, 79, 59.8, 17, 6.2, COLOR_RED_BG, COLOR_RED_BORDER,
          "BLOCCO TOTALE 24H!",
          "Circuit Breaker attivo:\nProtezione totale da crash",
          align='center', title_size=10, sub_size=8.5, lw=2)

draw_arrow(ax, 63.5, 59.8, 63.5, 56.5, color='#10B981', lw=1.5, label="NO", label_pos=0.5, label_bg='#064E3B', label_color='#A7F3D0', mut_scale=10)

# Filtro 1: Trend EMA 200
draw_card(ax, 52, 49.8, 23, 6.2, '#1E293B', '#38BDF8',
          "Filtro 1: Trend di Fondo",
          "Il prezzo è sopra la media\ndegli ultimi 33 giorni (EMA 200)?",
          align='center', title_size=10, sub_size=8.5, lw=1.5)

draw_arrow(ax, 75, 52.9, 79, 52.9, color='#64748B', lw=1.5, label="NO", label_pos=0.5, label_bg='#1E293B', label_color='#94A3B8', mut_scale=8)

draw_card(ax, 79, 49.8, 17, 6.2, COLOR_GRAY_BG, COLOR_GRAY_BORDER,
          "RESTA IN EURO",
          "Mercato in discesa/neutro:\nVietato comprare",
          align='center', title_size=9.5, sub_size=8.5, lw=1.5)

draw_arrow(ax, 63.5, 49.8, 63.5, 46.5, color='#10B981', lw=1.5, label="SÌ", label_pos=0.5, label_bg='#064E3B', label_color='#A7F3D0', mut_scale=10)

# Filtro 2: Rottura Massimi Donchian
draw_card(ax, 52, 39.8, 23, 6.2, '#1E293B', '#38BDF8',
          "Filtro 2: Esplosione Prezzo",
          "Il prezzo ha rotto il massimo\ndegli ultimi 3.5 giorni (20 candele)?",
          align='center', title_size=10, sub_size=8.5, lw=1.5)

draw_arrow(ax, 75, 42.9, 79, 42.9, color='#64748B', lw=1.5, label="NO", label_pos=0.5, label_bg='#1E293B', label_color='#94A3B8', mut_scale=8)

draw_card(ax, 79, 39.8, 17, 6.2, COLOR_GRAY_BG, COLOR_GRAY_BORDER,
          "RESTA IN EURO",
          "Nessuna rottura rialzista:\nPrezzo in consolidamento",
          align='center', title_size=9.5, sub_size=8.5, lw=1.5)

draw_arrow(ax, 63.5, 39.8, 63.5, 36.5, color='#10B981', lw=1.5, label="SÌ", label_pos=0.5, label_bg='#064E3B', label_color='#A7F3D0', mut_scale=10)

# Filtro 3: Volumi
draw_card(ax, 52, 29.8, 23, 6.2, '#1E293B', '#38BDF8',
          "Filtro 3: Spinta dei Volumi",
          "I volumi sono almeno 1.5 volte\nsuperiori alla media normale?",
          align='center', title_size=10, sub_size=8.5, lw=1.5)

draw_arrow(ax, 75, 32.9, 79, 32.9, color='#64748B', lw=1.5, label="NO", label_pos=0.5, label_bg='#1E293B', label_color='#94A3B8', mut_scale=8)

draw_card(ax, 79, 29.8, 17, 6.2, COLOR_GRAY_BG, COLOR_GRAY_BORDER,
          "RESTA IN EURO",
          "Falso movimento:\nSenza volumi manca la forza",
          align='center', title_size=9.5, sub_size=8.5, lw=1.5)

draw_arrow(ax, 63.5, 29.8, 63.5, 25.5, color='#10B981', lw=2, label="TUTTI SÌ", label_pos=0.5, label_bg='#064E3B', label_color='#A7F3D0', mut_scale=10)

# Esito B: Compra con Ordine Limit
draw_card(ax, 52, 14.5, 44, 10.5, '#064E3B', '#10B981',
          "TUTTI I FILTRI SUPERATI: COMPRA CON ORDINE LIMIT!",
          "1. Calcola la quantità esatta in base alla volatilità (Rischio max 4% = 20€ su 500€)\n2. Freno da Drawdown: Se vieni da 2 o 3 stop di fila, riduci la puntata (-30% / -50%)\n3. Invia Ordine Limit al prezzo esatto (Commissione minima 0.02% invece di 0.07%)\n4. Fissa subito lo Stop Loss iniziale a -2.2× Volatilità sotto l'ingresso",
          align='left', title_size=11, sub_size=9.2, lw=2.5)

# ----------------- FOOTER / BANNER INFERIORE -----------------
draw_card(ax, 4, 3.5, 92, 7.5, '#111827', '#374151',
          "I 3 PILASTRI MATEMATICI DEL BOT (PERCHÉ NON BRUCIA IL CONTO)",
          "1. Taglio rapido delle perdite: Lo Stop Loss a -2.2× ATR non viene mai violato.\n2. Trailing Profit infinito: Più la moneta sale, più la rete Chandelier sale, blindando i guadagni.\n3. Zero FOMO: Se mancano i volumi o il trend è ribassista, il bot resta serenamente liquido al 100% in Euro.",
          align='left', title_size=10.5, sub_size=9.2, lw=1.5)

# Output paths
out_p1 = "/Users/gi/teamwork_projects/trading_bot/albero_decisionale_bot.png"
out_p2 = "/Users/gi/.gemini/antigravity/brain/0afb07eb-994e-41ee-b3f9-bfd1f0c4e48c/albero_decisionale_bot.png"

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
plt.savefig(out_p1, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
plt.savefig(out_p2, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
print(f"Salvati con successo in {out_p1} e {out_p2}")
