"""Small SVG helpers shared by the README figures: themes, text and number formatting.

Each figure is written twice, for GitHub's light and dark themes, from the same
drawing code; README.md picks one with <picture> and prefers-color-scheme.
"""
from html import escape

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

THEMES = {
    'light': {'surface': '#ffffff', 'ink': '#0b0b0b', 'secondary': '#52514e', 'muted': '#8a8984',
              'grid': '#e6e5e1', 'band': '#f4f3f0', 'band2': '#ffffff', 'card': '#f6f8fa', 'card_edge': '#d0d7de',
              'series': ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'],
              'good': '#0ca30c', 'warning': '#fab219', 'serious': '#ec835a', 'critical': '#d03b3b'},
    'dark': {'surface': '#0d1117', 'ink': '#ffffff', 'secondary': '#c3c2b7', 'muted': '#8b8a83',
             'grid': '#30302d', 'band': '#161b22', 'band2': '#0d1117', 'card': '#161b22', 'card_edge': '#30363d',
             'series': ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
             'good': '#0ca30c', 'warning': '#fab219', 'serious': '#ec835a', 'critical': '#d03b3b'},
}


def text(x, y, s, size=13, fill='#000', weight=400, anchor='start', extra=''):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" {extra}>{escape(s)}</text>')


def svg(width, height, body, background):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img">'
            f'<rect width="{width}" height="{height}" fill="{background}"/>{"".join(body)}</svg>\n')


def polyline(points, stroke, width=2, extra=''):
    path = ' '.join(f'{x:.1f},{y:.1f}' for x, y in points)
    return (f'<polyline points="{path}" fill="none" stroke="{stroke}" stroke-width="{width}" '
            f'stroke-linejoin="round" stroke-linecap="round" {extra}/>')
