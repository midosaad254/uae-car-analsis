# install_deps.py
import importlib
import subprocess
import sys

# قايمة المكتبات المطلوبة
required_libraries = [
    'dash',
    'pandas',
    'plotly',
    'sklearn',  # scikit-learn بيتم تثبيته باسم sklearn
    'joblib',
    'statsmodels',
    'textblob',
    'dash_leaflet'  # المكتبة اللي كانت ناقصة
]

def check_and_install(library):
    """تشييك لو المكتبة موجودة وتثبيتها لو مش موجودة."""
    try:
        importlib.import_module(library)
        print(f"المكتبة '{library}' موجودة بالفعل.")
    except ImportError:
        print(f"المكتبة '{library}' مش موجودة، جاري التثبيت...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", library])
            print(f"تم تثبيت '{library}' بنجاح!")
        except subprocess.CalledProcessError as e:
            print(f"فشل تثبيت '{library}': {e}")
            sys.exit(1)

if __name__ == "__main__":
    print("جاري التشييك على المكتبات المطلوبة...\n")
    for lib in required_libraries:
        check_and_install(lib)
    print("\nكل المكتبات المطلوبة تم التشييك عليها أو تثبيتها!")
    print("جربي تشغيل 'python app.py' دلوقتي.")