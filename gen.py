import os
content = open('src/ui/main_window_new.py', 'r', encoding='utf-8').read() if os.path.exists('src/ui/main_window_new.py') else 'empty'
print(len(content))
