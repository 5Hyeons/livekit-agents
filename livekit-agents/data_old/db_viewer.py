#!/usr/bin/env python3
"""
SQLite Database Viewer for LiveKit Agents Chat History
Usage: python db_viewer.py [database_file]
"""

import sqlite3
import json
import sys
import os
from datetime import datetime
from pathlib import Path


def view_database(db_path):
    """View contents of LiveKit Agents SQLite database"""
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' not found")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"=== DATABASE: {os.path.basename(db_path)} ===\n")
        
        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"Tables: {[table[0] for table in tables]}\n")
        
        # Show user info
        print('=== USER INFO ===')
        cursor.execute('SELECT * FROM users')
        users = cursor.fetchall()
        
        if not users:
            print("No users found")
        else:
            # Get column names
            cursor.execute("PRAGMA table_info(users)")
            columns_info = cursor.fetchall()
            column_names = [col[1] for col in columns_info]
            
            for user in users:
                # Create dict from row for easier access
                user_dict = dict(zip(column_names, user))
                
                print(f'Participant ID: {user_dict["participant_id"]}')
                print(f'Display Name: {user_dict.get("display_name", "N/A")}') 
                print(f'First Seen: {user_dict.get("first_seen", "N/A")}')
                print(f'Last Seen: {user_dict.get("last_seen", "N/A")}')
                print(f'Language: {user_dict.get("language", "N/A")}')
                
                # Token information
                if 'remaining_tokens' in user_dict:
                    print(f'💰 Token Balance:')
                    print(f'  - Remaining: {user_dict.get("remaining_tokens", 0)} tokens')
                    print(f'  - Used: {user_dict.get("total_tokens_used", 0)} tokens')
                    print(f'  - Granted: {user_dict.get("total_tokens_granted", 2000)} tokens')
                
                # Metadata
                metadata_str = user_dict.get("metadata")
                if metadata_str:
                    try:
                        metadata = json.loads(metadata_str)
                        print(f'Metadata: {json.dumps(metadata, indent=2, ensure_ascii=False)}')
                    except:
                        print(f'Metadata: {metadata_str}')
                print()
        
        # Show usage metrics summary
        print('=== USAGE METRICS SUMMARY ===')
        
        # Check if usage_metrics table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='usage_metrics'")
        if cursor.fetchone():
            # LLM usage summary
            cursor.execute('''
                SELECT 
                    COUNT(*) as request_count,
                    SUM(prompt_tokens) as total_prompt_tokens,
                    SUM(prompt_cached_tokens) as total_cached_tokens,
                    SUM(completion_tokens) as total_completion_tokens,
                    SUM(total_tokens) as total_tokens,
                    AVG(duration) as avg_duration
                FROM usage_metrics 
                WHERE metric_type = 'llm'
            ''')
            llm_summary = cursor.fetchone()
            
            print('LLM Usage:')
            if llm_summary[0] > 0:  # Has LLM usage
                print(f'  Total Requests: {llm_summary[0]}')
                print(f'  Total Tokens: {llm_summary[4] or 0}')
                print(f'    - Prompt: {llm_summary[1] or 0}')
                print(f'    - Cached: {llm_summary[2] or 0}')
                print(f'    - Completion: {llm_summary[3] or 0}')
                print(f'  Avg Duration: {llm_summary[5]:.2f}s' if llm_summary[5] else '  Avg Duration: N/A')
            else:
                print('  No LLM usage recorded')
            
            # TTS usage summary
            cursor.execute('''
                SELECT 
                    COUNT(*) as request_count,
                    SUM(characters_count) as total_characters,
                    SUM(audio_duration) as total_audio_duration,
                    AVG(duration) as avg_duration
                FROM usage_metrics 
                WHERE metric_type = 'tts'
            ''')
            tts_summary = cursor.fetchone()
            
            print('\nTTS Usage:')
            if tts_summary[0] > 0:  # Has TTS usage
                print(f'  Total Requests: {tts_summary[0]}')
                print(f'  Total Characters: {tts_summary[1] or 0}')
                print(f'  Total Audio Duration: {tts_summary[2]:.2f}s' if tts_summary[2] else '  Total Audio Duration: 0s')
                print(f'  Avg Processing Duration: {tts_summary[3]:.2f}s' if tts_summary[3] else '  Avg Processing Duration: N/A')
            else:
                print('  No TTS usage recorded')
            
            # Show recent usage details
            print('\n=== RECENT USAGE DETAILS (Last 10) ===')
            cursor.execute('''
                SELECT metric_type, timestamp, request_id,
                       prompt_tokens, completion_tokens, total_tokens,
                       characters_count, audio_duration, duration
                FROM usage_metrics 
                ORDER BY timestamp DESC
                LIMIT 10
            ''')
            recent_usage = cursor.fetchall()
            
            if recent_usage:
                for usage in recent_usage:
                    metric_type, timestamp, request_id, prompt_tokens, completion_tokens, total_tokens, chars, audio_dur, duration = usage
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        readable_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        readable_time = timestamp
                    
                    if metric_type == 'llm':
                        print(f'[{readable_time}] LLM: {total_tokens or 0} tokens (prompt: {prompt_tokens or 0}, completion: {completion_tokens or 0}), duration: {duration:.2f}s')
                    elif metric_type == 'tts':
                        print(f'[{readable_time}] TTS: {chars or 0} chars, {audio_dur:.2f}s audio, duration: {duration:.2f}s')
            else:
                print('No usage records found')
            print()
        else:
            print('No usage metrics table found (older database version)')
            print()
        
        # Show chat history
        print('=== CHAT HISTORY ===')
        cursor.execute('SELECT * FROM chat_history ORDER BY timestamp')
        messages = cursor.fetchall()
        
        print(f'Total messages: {len(messages)}')
        
        if messages:
            print('\nShowing last 20 messages:')
            print('-' * 50)
            # Show only last 20 messages for readability
            for msg in messages[-20:]:
                msg_id, participant_id, session_id, timestamp, role, content, interrupted = msg
                # Convert timestamp to readable format
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    readable_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    readable_time = timestamp
                
                interrupted_text = ' (INTERRUPTED)' if interrupted else ''
                print(f'[{readable_time}] {role.upper()}{interrupted_text}:')
                # Truncate long messages for readability
                if len(content) > 200:
                    print(f'  {content[:200]}...')
                else:
                    print(f'  {content}')
                print()
        else:
            print("No chat messages found")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Error: {e}")


def list_databases(data_dir):
    """List all SQLite database files in the data directory"""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Directory '{data_dir}' not found")
        return
    
    db_files = list(data_path.glob("*.db"))
    if not db_files:
        print(f"No .db files found in '{data_dir}'")
        return
    
    print(f"Found {len(db_files)} database file(s):")
    for i, db_file in enumerate(db_files, 1):
        print(f"{i}. {db_file.name}")
    
    return db_files


def main():
    if len(sys.argv) > 1:
        # Specific database file provided
        db_path = sys.argv[1]
        view_database(db_path)
    else:
        # No specific file, list available databases in current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_files = list_databases(current_dir)
        
        if db_files:
            print("\nSelect a database to view:")
            try:
                choice = int(input("Enter number (or 0 to exit): "))
                if choice == 0:
                    return
                elif 1 <= choice <= len(db_files):
                    print("\n" + "="*50 + "\n")
                    view_database(str(db_files[choice-1]))
                else:
                    print("Invalid choice")
            except (ValueError, KeyboardInterrupt):
                print("\nExiting...")


if __name__ == "__main__":
    main()