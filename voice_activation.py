import speech_recognition as sr

def ouvir_microfone():
    # Inicializa o reconhecedor
    reconhecedor = sr.Recognizer()

    with sr.Microphone() as source:
        # Ajusta o ruído ambiente
        reconhecedor.adjust_for_ambient_noise(source, duration=1)
        print("R.A.P.T.O.R: Estou ouvindo senhor")
        
        # Ouve o áudio
        audio = reconhecedor.listen(source)

    try:
        # Tenta reconhecer o que foi dito (em Português)
        texto = reconhecedor.recognize_google(audio, language='pt-BR')
        print(f"Você disse: {texto}")
        return texto

    except sr.UnknownValueError:
        print("Não entendi o que você disse.")
    except sr.RequestError:
        print("Erro ao acessar o serviço de reconhecimento.")

    if "A" in texto:
        print("Executando R.A.P.T.O.R")

# Executa a função
ouvir_microfone()