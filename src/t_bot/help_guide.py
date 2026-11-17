"""Shared user guide for Telegram commands and ADK capabilities."""

from collections.abc import Iterable

from t_bot.manifest import Manifest
from t_bot.tool_names import BASIC_TOOL_DESCRIPTIONS

COMMAND_GUIDE = """Команди Telegram (надсилайте окремим повідомленням):
/start — початок роботи та довідка.
/help — ця довідка; також напишіть «допомога» або «help».
/status — готовність бота та ID ваших останніх задач.
/builds — останні конвеєри створення розширень.
/build BUILD_ID — збережений звіт, ТЗ, код, QA-тести й зауваження конвеєра.
/models — список установлених моделей Ollama.
/model — поточна модель головного агента.
/model MODEL [AGENT_ID] — вибрати встановлену модель; без AGENT_ID — root.
Приклад: /model gemma4:12b. Зміна діє з наступного повідомлення.
/agents — активні агенти та субагенти, їхні ID і батьківські агенти.
/tools — базові інструменти та активні інструменти-розширення.
/extensions — усі збережені версії розширень та стан активації.
/source ID [VERSION] — Python-код і маніфест розширення.
/extension enable|disable|delete|rollback ID [VERSION] — керування розширенням.
Приклади: /extension disable word_counter; /extension rollback word_counter 1.
/cancel TASK_ID — скасувати власну задачу; ID дивіться в /status.
/new — нова сесія в поточному чаті; спершу завершіть або скасуйте задачі.
/restart — керовано перезапустити бота зі збереженням стану.
/screenshot — скріншот комп’ютера бота в цей чат.
Також: «Що зараз на екрані» або «Що на екрані». Потрібен доступ ОС до екрана.
/time — поточна дата, час і налаштований часовий пояс.
Також: «Котра година» або «Скока время»."""

TOOL_EXAMPLES = {
    "install_dependencies": "Встанови matplotlib для extensions. "
    "pip сам визначить повний набір залежностей; передайте результат до create_extension.",
    "send_file": "Надішли reports/report.pdf вкладенням у цей чат. "
    "Шлях відносний до FILESYSTEM_ROOT; до 50 МБ або меншого ATTACHMENT_MAX_BYTES.",
    "list_files": "Покажи файли в каталозі reports.",
    "read_file": "Прочитай reports/notes.txt; для великого файлу попросіть наступну частину.",
    "write_file": "Запиши текст «Привіт» у notes.txt. Наявний вміст буде замінено.",
    "create_directory": "Створи каталог reports/2026.",
    "delete_file": "Видали reports/old.txt. Файл буде видалено з диска.",
    "search_web": "Знайди в інтернеті документацію Python sqlite3, дай 5 посилань.",
    "list_databases": "Покажи доступні SQLite-бази та їхні ідентифікатори.",
    "create_database": "Створи SQLite-базу notes.",
    "database_schema": "Покажи структуру бази notes (ID візьми зі списку баз).",
    "table_schema": "Покажи колонки, ключі й індекси таблиці items у базі notes.",
    "execute_sql": "У базі notes виконай SELECT * FROM items LIMIT 10; "
    "для значень використовуй параметри. state/session доступні лише для читання.",
    "execute_sql_script": "У базі notes створи таблицю items(id INTEGER PRIMARY KEY, "
    "text TEXT), потім додай рядок. Користувацькі бази дозволяють зміну даних.",
    "run_command": "Виконай команду pwd у робочому каталозі; покажи stdout, stderr "
    "і код завершення. Запис обмежений дозволеним робочим простором, час виконання обмежений.",
    "take_screenshot": "Що на екрані? Скріншот надсилається в чат, а не моделі.",
    "get_current_datetime": "Котра година? Повертає поточні дату, час і часовий пояс.",
    "get_help": "Допомога. Повертає актуальні команди, інструменти й приклади.",
}


def render_help(manifests: Iterable[Manifest] = ()) -> str:
    """Render all built-ins and the supplied active extension manifests without model calls."""
    sections = [
        "Довідка бота\nНадсилайте звичайним текстом завдання для ADK-агента. "
        "Назви інструментів нижче — не slash-команди: агент викликає їх за вашим запитом.",
        COMMAND_GUIDE,
        "Базові інструменти (приклади повідомлень):\n"
        + "\n".join(
            f"{name} — {description}.\nПриклад: {TOOL_EXAMPLES[name]}"
            for name, description in BASIC_TOOL_DESCRIPTIONS.items()
        ),
        "Керування через агента:\n"
        "list_resources — перелічити моделі, задачі, розширення й інструменти.\n"
        "set_model — вибрати встановлену модель для root або агента.\n"
        "restart_bot — перезапустити бота; приклад: «Перезапусти себе».\n"
        "create_extension — створити Python-інструмент, агента чи субагента з ADK. "
        "Передайте dependencies із назвами основних бібліотек; pip визначить решту. "
        "Інтернет доступний extensions за замовчуванням. "
        "Для графіка з word_count використайте input_from=word_count; "
        "передавайте його char_frequency інструменту малювання.\n"
        "Приклад: «Створи інструмент word_counter для підрахунку слів у тексті».\n"
        "Для агента вкажіть роль, модель зі /models і, для субагента, батьківський ID.\n"
        "manage_extension — активувати, вимкнути, видалити або відкотити розширення.\n"
        "Нові розширення підключаються з наступного повідомлення. "
        "Файлові шляхи відносні до FILESYSTEM_ROOT. "
        "Інструменти змінюють дані лише в межах налаштованих дозволів.",
    ]
    extensions = [
        f"{manifest.id} v{manifest.version} ({manifest.kind}, parent={manifest.parent}) — "
        f"{manifest.description}\nЯк користуватися: {manifest.example_usage}"
        for manifest in manifests
    ]
    sections.append(
        "Активні розширення (субагентів викликає їхній батьківський агент):\n"
        + ("\n\n".join(extensions) or "Поки немає. Попросіть агента створити потрібне.")
    )
    return "\n\n".join(sections)
