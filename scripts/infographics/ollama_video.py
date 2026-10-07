from pathlib import Path

from board import Board

root = Path(__file__).resolve().parents[2]
folder = root / 'static/img/genai/ollama'
folder.mkdir(parents=True, exist_ok=True)

b = Board(960, 470, 'One service, several ways to ask', 'Follow the request from your interface to the model that performs inference.')
b.group(20, 98, 920, 345, 'On your computer', 'blue')
a = b.card(45, 150, 230, 110, 'Your interface', ['CLI / Python / REST', 'LangChain / desktop'], 'blue', size=15)
s = b.card(360, 150, 235, 110, 'Ollama service', ['localhost:11434', 'Receives the request'], 'purple', size=15)
m = b.card(680, 150, 230, 110, 'Local model', ['Loaded in RAM / VRAM', 'CPU / GPU generate text'], 'green', size=14)
d = b.card(680, 315, 230, 95, 'Downloaded files', ['Stored on disk', 'Loaded when needed'], 'orange', size=14)
b.arrow(a.right(), s.left(), label='request', color='blue')
b.arrow(s.right(), m.left(), label='inference', color='purple')
b.arrow(d.top(), m.bottom(), label='load', color='orange', label_dx=38)
b.text(360, 330, 'A cloud model uses remote hardware.', size=14, anchor='middle', color='grey')
b.text(360, 360, 'The prompt then leaves your computer.', size=14, anchor='middle', color='grey')
b.save(folder / 'interfaces-and-service.svg')

b = Board(960, 520, 'The laptop price needs two real tool results', 'A requested function and an executed function are different events.')
xs = [35, 355, 675]
q = b.card(xs[0], 105, 250, 110, '1. Inventory request', ['product = laptop', 'Application runs lookup'], 'blue', size=15)
i = b.card(xs[1], 105, 250, 110, '2. Inventory result', ['stock = 5', 'base price = 1200'], 'green', size=15)
d = b.card(xs[2], 105, 250, 110, '3. Discount request', ['base price = 1200', 'customer years = 5'], 'purple', size=15)
f = b.card(xs[2], 305, 250, 130, '4. Python computes', ['5 x 5% = 25%', '1200 x 0.75 = 900', 'Result returns to model'], 'orange', size=15)
a = b.card(xs[1], 305, 250, 130, '5. Model answers', ['History includes 900', 'Price comes from Python', 'Stock comes from lookup'], 'green', size=15)
stop = b.card(xs[0], 305, 250, 130, 'Stop after inventory?', ['No discount executed', 'A price in generated text', 'is not a tool result'], 'red', size=14)
b.arrow(q.right(), i.left(), color='blue')
b.arrow(i.right(), d.left(), label='next request', color='purple')
b.arrow(d.bottom(), f.top(), label='execute', color='purple', label_dx=40)
b.arrow(f.left(), a.right(), label='900', color='green')
b.arrow(i.bottom(0.2), stop.top(0.8), color='red', dashed=True)
b.text(480, 483, 'Five years gives 25% off: 1200 x 0.75 = 900. Inventory alone does not calculate it.', size=14, color='grey')
b.save(folder / 'shop-tool-results.svg')
