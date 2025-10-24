import runpy
print(">>> driver: running run_path", flush=True)
runpy.run_path(r'.\paper_trader.py', run_name='__main__')
print(">>> driver: done", flush=True)
