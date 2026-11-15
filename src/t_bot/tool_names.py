"""Stable built-in tool names, shared by discovery and extension validation."""

BASIC_TOOL_DESCRIPTIONS = {
    "send_file": "Надсилання файлу з FILESYSTEM_ROOT вкладенням у поточний чат",
    "get_help": "Допомога / help: усі команди, інструменти та гайд із прикладами",
    "list_files": "Перегляд файлів і каталогів у FILESYSTEM_ROOT",
    "read_file": "Читання текстового UTF-8 файлу",
    "write_file": "Створення або заміна текстового UTF-8 файлу",
    "create_directory": "Створення каталогу",
    "delete_file": "Видалення звичайного файлу",
    "search_web": "Пошук в інтернеті з посиланнями й короткими описами",
    "list_databases": "Перегляд SQLite-баз: службові read-only та користувацькі",
    "create_database": "Створення користувацької SQLite-бази з повним доступом",
    "database_schema": "Структура бази: таблиці, view, індекси й тригери",
    "table_schema": "Структура таблиці: колонки, ключі й індекси",
    "execute_sql": "SQL-запит; state/session лише читання, користувацькі бази — повний доступ",
    "execute_sql_script": "SQL-скрипт із кількох команд для користувацької бази",
    "run_command": "Консольні команди: stdout, stderr, код завершення й таймаут",
    "take_screenshot": "Що на екрані: скріншот комп’ютера бота в поточний чат",
    "get_current_datetime": "Котра година / Скока время: поточна дата, час і часовий пояс",
}
