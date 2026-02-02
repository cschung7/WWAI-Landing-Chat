#!/bin/bash
# Start WWAI Landing Chat Backend

# Load environment variables
if [ -f /mnt/nas/gpt/.env ]; then
    # Export without quotes
    while IFS='=' read -r key value; do
        if [[ ! $key =~ ^# && -n $key ]]; then
            value=$(echo "$value" | tr -d '"' | tr -d "'")
            export "$key=$value"
        fi
    done < /mnt/nas/gpt/.env
fi

cd "$(dirname "$0")"
python3 -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8091}
