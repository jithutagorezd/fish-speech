import sys
with open('D:/Projects/ai_audio/fish_audio/fish-speech/webui_v2/app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

create_voice_tab_start = -1
create_voice_tab_end = -1
for i, line in enumerate(lines):
    if 'with gr.Tab("➕ Create Voice"):' in line:
        create_voice_tab_start = i
        break

if create_voice_tab_start != -1:
    for i in range(create_voice_tab_start + 1, len(lines)):
        if 'gr.HTML(FOOTER_HTML)' in lines[i]:
            create_voice_tab_end = i
            break

if create_voice_tab_start != -1 and create_voice_tab_end != -1:
    tab_lines = lines[create_voice_tab_start:create_voice_tab_end]
    
    dispatch_line = -1
    for i, line in enumerate(lines):
        if 'def dispatch(' in line:
            dispatch_line = i
            break
            
    if dispatch_line != -1:
        del lines[create_voice_tab_start:create_voice_tab_end]
        lines = lines[:dispatch_line] + tab_lines + ['\n'] + lines[dispatch_line:]
        with open('D:/Projects/ai_audio/fish_audio/fish-speech/webui_v2/app.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print('Successfully moved the tab!')
    else:
        print('Could not find dispatch function')
else:
    print('Could not find Create Voice tab or end of tab')
