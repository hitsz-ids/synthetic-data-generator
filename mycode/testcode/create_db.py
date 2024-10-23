import sqlite3

def create_db(DB_NAME):
    # Connect to SQLite database
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Student (
        student_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        date_of_birth DATE NOT NULL,
        major TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Professor (
        professor_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        department TEXT NOT NULL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Course (
        course_id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        credits INTEGER NOT NULL,
        department TEXT NOT NULL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Department (
        department_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        building TEXT NOT NULL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Enrollment (
        enrollment_id INTEGER PRIMARY KEY,
        student_id INTEGER NOT NULL,
        course_id INTEGER NOT NULL,
        semester TEXT NOT NULL,
        grade TEXT,
        FOREIGN KEY(student_id) REFERENCES Student(student_id),
        FOREIGN KEY(course_id) REFERENCES Course(course_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Assignment (
        assignment_id INTEGER PRIMARY KEY,
        course_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        due_date DATE NOT NULL,
        FOREIGN KEY(course_id) REFERENCES Course(course_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Submission (
        submission_id INTEGER PRIMARY KEY,
        assignment_id INTEGER NOT NULL,
        student_id INTEGER NOT NULL,
        submission_date DATE NOT NULL,
        grade TEXT,
        FOREIGN KEY(assignment_id) REFERENCES Assignment(assignment_id),
        FOREIGN KEY(student_id) REFERENCES Student(student_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Schedule (
        schedule_id INTEGER PRIMARY KEY,
        course_id INTEGER NOT NULL,
        professor_id INTEGER NOT NULL,
        room TEXT NOT NULL,
        time_slot TEXT NOT NULL,
        FOREIGN KEY(course_id) REFERENCES Course(course_id),
        FOREIGN KEY(professor_id) REFERENCES Professor(professor_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Major (
        major_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        department_id INTEGER NOT NULL,
        FOREIGN KEY(department_id) REFERENCES Department(department_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Textbook (
        textbook_id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        author TEXT NOT NULL,
        isbn TEXT UNIQUE NOT NULL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS CourseTextbook (
        course_id INTEGER NOT NULL,
        textbook_id INTEGER NOT NULL,
        PRIMARY KEY(course_id, textbook_id),
        FOREIGN KEY(course_id) REFERENCES Course(course_id),
        FOREIGN KEY(textbook_id) REFERENCES Textbook(textbook_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Library (
        library_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        location TEXT NOT NULL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Book (
        book_id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        author TEXT NOT NULL,
        isbn TEXT UNIQUE NOT NULL,
        library_id INTEGER NOT NULL,
        FOREIGN KEY(library_id) REFERENCES Library(library_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS BookLoan (
        loan_id INTEGER PRIMARY KEY,
        book_id INTEGER NOT NULL,
        student_id INTEGER NOT NULL,
        loan_date DATE NOT NULL,
        return_date DATE,
        FOREIGN KEY(book_id) REFERENCES Book(book_id),
        FOREIGN KEY(student_id) REFERENCES Student(student_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ResearchGroup (
        group_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        focus_area TEXT NOT NULL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ResearchProject (
        project_id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        group_id INTEGER NOT NULL,
        start_date DATE NOT NULL,
        end_date DATE,
        FOREIGN KEY(group_id) REFERENCES ResearchGroup(group_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ProjectMember (
        project_id INTEGER NOT NULL,
        professor_id INTEGER NOT NULL,
        PRIMARY KEY(project_id, professor_id),
        FOREIGN KEY(project_id) REFERENCES ResearchProject(project_id),
        FOREIGN KEY(professor_id) REFERENCES Professor(professor_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Lab (
        lab_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        building TEXT NOT NULL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS LabEquipment (
        equipment_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        lab_id INTEGER NOT NULL,
        FOREIGN KEY(lab_id) REFERENCES Lab(lab_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS EquipmentMaintenance (
        maintenance_id INTEGER PRIMARY KEY,
        equipment_id INTEGER NOT NULL,
        date DATE NOT NULL,
        details TEXT NOT NULL,
        FOREIGN KEY(equipment_id) REFERENCES LabEquipment(equipment_id)
    )
    ''')

    # Commit changes and close the connection
    conn.commit()
    conn.close()
