"""
Main launcher for PanelCleaner Server with interactive CLI menu
Run this script to start the entire server system
"""
import os
import sys
import subprocess
import asyncio
import secrets
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from typing import Optional

# Configure logging
logger.remove()
logger.add(sys.stderr, format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# Load environment variables
load_dotenv()


class InteractiveMenu:
    """Interactive CLI menu for server configuration"""
    
    def __init__(self):
        self.config_file = ".env"
        self.api_key = os.getenv("DISCORD_BOT_API_KEY", self._generate_api_key())
    
    def display_main_menu(self):
        """Display main menu options"""
        while True:
            print("\n" + "=" * 70)
            print("🖥️  PanelCleaner Server - Main Menu".center(70))
            print("=" * 70)
            print("\n  1. 🚀 تشغيل الخادم (Start Server)")
            print("  2. ⚙️  الإعدادات (Settings)")
            print("  3. 📊 حالة النظام (System Status)")
            print("  4. ❌ خروج (Exit)\n")
            
            choice = input("اختر خيار (Choose option): ").strip()
            
            if choice == "1":
                self.start_server()
            elif choice == "2":
                self.settings_menu()
            elif choice == "3":
                self.system_status()
            elif choice == "4":
                print("\n👋 إلى اللقاء! (Goodbye!)")
                sys.exit(0)
            else:
                print("❌ خيار غير صحيح (Invalid choice)")
    
    def settings_menu(self):
        """Settings submenu"""
        while True:
            print("\n" + "=" * 70)
            print("⚙️  الإعدادات (Settings)".center(70))
            print("=" * 70)
            print("\n  1. 🔑 إعادة إنشاء مفتاح API (Regenerate API Key)")
            print("  2. 📝 تحديث بيانات Supabase (Update Supabase Credentials)")
            print("  3. 🔗 تحديث بيانات Google Drive (Update Google Drive Credentials)")
            print("  4. ⬅️  العودة (Back)\n")
            
            choice = input("اختر خيار (Choose option): ").strip()
            
            if choice == "1":
                self.regenerate_api_key()
            elif choice == "2":
                self.update_supabase()
            elif choice == "3":
                self.update_google_drive()
            elif choice == "4":
                break
            else:
                print("❌ خيار غير صحيح (Invalid choice)")
    
    def regenerate_api_key(self):
        """Generate and save new API key"""
        print("\n" + "-" * 70)
        new_key = self._generate_api_key()
        self._save_env_variable("DISCORD_BOT_API_KEY", new_key)
        
        print(f"\n✅ تم إنشاء مفتاح API جديد (New API Key generated)")
        print(f"\n🔑 المفتاح الجديد (New Key):")
        print(f"   {new_key}\n")
        print("⚠️  احفظ هذا المفتاح وأرسله لصديقك في البوت (Save this key and share with your Discord Bot)\n")
        
        input("اضغط Enter للمتابعة (Press Enter to continue)...")
    
    def update_supabase(self):
        """Update Supabase credentials"""
        print("\n" + "-" * 70)
        print("\n📝 تحديث بيانات Supabase (Update Supabase Credentials)")
        print("\nاترك الحقل فارغاً لتخطيه (Leave empty to skip)\n")
        
        url = input("Supabase URL: ").strip()
        if url:
            self._save_env_variable("SUPABASE_URL", url)
        
        key = input("Supabase API Key: ").strip()
        if key:
            self._save_env_variable("SUPABASE_KEY", key)
        
        print("\n✅ تم حفظ البيانات (Credentials saved)")
        input("اضغط Enter للمتابعة (Press Enter to continue)...")
    
    def update_google_drive(self):
        """Update Google Drive credentials"""
        print("\n" + "-" * 70)
        print("\n🔗 تحديث بيانات Google Drive (Update Google Drive Credentials)")
        print("\nاترك الحقل فارغاً لتخطيه (Leave empty to skip)\n")
        
        service_account_file = input("Google Service Account JSON File Path: ").strip()
        if service_account_file and os.path.exists(service_account_file):
            self._save_env_variable("GOOGLE_SERVICE_ACCOUNT_JSON", service_account_file)
            print("✅ تم حفظ الملف (File saved)")
        
        folder_id = input("Google Drive Output Folder ID: ").strip()
        if folder_id:
            self._save_env_variable("GOOGLE_OUTPUT_FOLDER_ID", folder_id)
        
        print("\n✅ تم حفظ البيانات (Credentials saved)")
        input("اضغط Enter للمتابعة (Press Enter to continue)...")
    
    def system_status(self):
        """Display system status information"""
        print("\n" + "-" * 70)
        print("📊 حالة النظام (System Status)")
        print("-" * 70)
        
        print(f"\n✅ API Key: {'مكون' if os.getenv('DISCORD_BOT_API_KEY') else '❌ غير مكون'}")
        print(f"✅ Supabase: {'مكون' if os.getenv('SUPABASE_URL') else '❌ غير مكون'}")
        print(f"✅ Google Drive: {'مكون' if os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON') else '❌ غير مكون'}")
        print(f"✅ Redis: {'مكون' if os.getenv('REDIS_URL') else '❌ غير مكون'}")
        
        print("\n" + "-" * 70)
        input("اضغط Enter للمتابعة (Press Enter to continue)...")
    
    def start_server(self):
        """Start the server"""
        print("\n" + "=" * 70)
        print("🚀 تشغيل الخادم (Starting Server)".center(70))
        print("=" * 70)
        print("\n✅ التحقق من المكتبات المطلوبة (Checking dependencies)...")
        
        # Check if requirements are installed
        if not self._check_dependencies():
            print("\n❌ المكتبات المطلوبة غير مثبتة (Required packages not installed)")
            response = input("هل تريد تثبيتها الآن؟ (Install now?) (y/n): ").strip().lower()
            if response == 'y':
                self._install_dependencies()
            else:
                return
        
        print("\n✅ جاهز للبدء (Ready to start)")
        print("\n🔄 تشغيل الخادم والـ Worker (Starting server and worker)...")
        
        try:
            # This would start the actual server
            # For now, we'll show the configuration
            self._show_server_config()
            
            print("\n💡 للبدء الفعلي، يجب تشغيل:")
            print("   uvicorn main:app --host 0.0.0.0 --port 10100 --reload")
            
            input("\nاضغط Enter للعودة (Press Enter to return)...")
        except KeyboardInterrupt:
            print("\n\n⚠️  تم إيقاف الخادم (Server stopped)")
    
    def _check_dependencies(self) -> bool:
        """Check if required packages are installed"""
        required_packages = [
            'fastapi', 'uvicorn', 'pydantic', 'loguru',
            'python-dotenv', 'pillow'
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing.append(package)
        
        if missing:
            print(f"❌ المكتبات المفقودة (Missing packages): {', '.join(missing)}")
            return False
        
        return True
    
    def _install_dependencies(self):
        """Install required packages"""
        print("\n📦 تثبيت المكتبات (Installing packages)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ تم تثبيت المكتبات (Packages installed)")
    
    def _show_server_config(self):
        """Display server configuration"""
        print("\n" + "-" * 70)
        print("⚙️  إعدادات الخادم (Server Configuration)")
        print("-" * 70)
        print(f"\n📌 API Host: {os.getenv('API_HOST', '0.0.0.0')}")
        print(f"📌 API Port: {os.getenv('API_PORT', '8000')}")
        print(f"📌 Debug Mode: {os.getenv('DEBUG', 'false')}")
        print(f"📌 API Key: {self.api_key[:10]}..." if self.api_key else "📌 API Key: Not configured")
        print("-" * 70)
    
    @staticmethod
    def _generate_api_key() -> str:
        """Generate a new secure API key"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def _save_env_variable(key: str, value: str):
        """Save environment variable to .env file"""
        env_file = Path(".env")
        
        # Read existing content
        content = ""
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                content = f.read()
        
        # Update or add variable
        lines = content.split('\n')
        found = False
        new_lines = []
        
        for line in lines:
            if line.startswith(f"{key}="):
                new_lines.append(f"{key}={value}")
                found = True
            else:
                new_lines.append(line)
        
        if not found:
            new_lines.append(f"{key}={value}")
        
        # Write back
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))


def main():
    """Main entry point"""
    # Ensure profiles directory exists
    profiles_dir = Path("pcleaner/profiles")
    profiles_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure temp chapters directory exists
    temp_chapters_dir = Path("temp_chapters")
    temp_chapters_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("🎉 مرحباً بك في PanelCleaner Server".center(70))
    print("=" * 70)
    
    menu = InteractiveMenu()
    menu.display_main_menu()


if __name__ == "__main__":
    main()
