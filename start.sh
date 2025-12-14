#!/bin/bash
# Script untuk menjalankan aplikasi dengan environment yang benar

# Pindah ke direktori script
cd "$(dirname "$0")"

# Cek apakah venv ada
if [ -d "venv" ]; then
    echo "Starting Sentiment Analysis System..."
    
    # Gunakan python dari venv secara langsung
    ./venv/bin/python app.py
else
    echo "Error: Virtual environment 'venv' tidak ditemukan."
    echo "Silakan buat venv terlebih dahulu atau install dependencies."
fi
