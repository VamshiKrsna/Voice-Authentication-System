from cx_Freeze import setup,Executable

setup(
    name= 'cosVoiceAuth',
    version= '1.0',
    description= 'Voice Authentication System',
    executables = [Executable("cosineapproach.py")]
)