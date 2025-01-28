from src import VoiceAuthenticator

def main():
    authenticator = VoiceAuthenticator()
    
    while True:
        print("\nVoice Authentication System")
        print("1. Enroll new user")
        print("2. Verify voice")
        print("3. Start authenticated transcription")
        print("4. Exit")
        
        choice = input("Choose an option: ")
        
        if choice == "1":
            user_id = input("Enter user ID: ")
            authenticator.train_model(user_id)
        
        elif choice == "2":
            user_id = input("Enter user ID to verify: ")
            if authenticator.verify_voice(user_id):
                print("✅ Voice verified!")
            else:
                print("❌ Voice verification failed!")
        
        elif choice == "3":
            user_id = input("Enter user ID: ")
            authenticator.transcribe_with_auth(user_id)
        
        elif choice == "4":
            break

if __name__ == "__main__":
    main()