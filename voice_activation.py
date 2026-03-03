import speech_recognition as sr
import subprocess

def ouvir_microfone():
    # Inicializa o reconhecedor fora do loop para economizar memória
    reconhecedor = sr.Recognizer()

    while True:
        with sr.Microphone() as source:
            # Ajusta o ruído ambiente para melhorar a precisão
            reconhecedor.adjust_for_ambient_noise(source, duration=1)
            print("R.A.P.T.O.R: Estou ouvindo, senhor...")
            
            # Ouve o áudio
            audio = reconhecedor.listen(source)

        try:
            # Tenta reconhecer o que foi dito
            texto = reconhecedor.recognize_google(audio, language='pt-BR')
            print(f"Você disse: {texto}")

            # Convertemos para minúsculas para aceitar "Olá", "ola", "OLA", etc.
            comando = texto.lower()

            # Verifica se a frase de ativação está no que foi dito
            if "ola raptor" in comando or "olá raptor" in comando:
                print(">>> Comando reconhecido! Executando R.A.P.T.O.R...")
                
                # Executa o script externo de forma limpa
                subprocess.run(["python", "raptor.py"])
                
                # Se você quiser que ele pare de ouvir após executar, use 'break' ou 'return'
                # break 

        except sr.UnknownValueError:
            # Caso o microfone capture barulho mas não identifique palavras
            print("R.A.P.T.O.R: Não entendi o comando.")
            
        except sr.RequestError:
            # Caso esteja sem internet ou o serviço do Google caia
            print("R.A.P.T.O.R: Erro de conexão com o serviço de reconhecimento.")

# Executa a função
if __name__ == "__main__":
    ouvir_microfone()