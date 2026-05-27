import runpy
import traceback

try:
    runpy.run_path('src/evaluate.py', run_name='__main__')
except Exception:
    with open('eval_traceback.txt', 'w', encoding='utf-8') as f:
        traceback.print_exc(file=f)
    raise
