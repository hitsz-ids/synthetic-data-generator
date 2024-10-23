from tqdm import notebook as tqdm
import sqlite3
import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()
fake_u = Faker().unique


# Function to generate random dates
def random_date(start, end):
    return start + timedelta(
        seconds=random.randint(0, int((end - start).total_seconds())))


def _3(TOTAL_COUNT, conn, cursor):
    cursor.execute("SELECT assignment_id FROM Assignment")
    assignments = cursor.fetchall()

    cursor.execute("SELECT student_id FROM Student")
    students = cursor.fetchall()

    # Convert fetched data into more usable formats (lists of ids)
    assignment_ids = [assignment[0] for assignment in assignments]
    student_ids = [student[0] for student in students]

    start_date = datetime.strptime('2022-01-01', '%Y-%m-%d')
    end_date = datetime.strptime('2022-12-31', '%Y-%m-%d')

    submission_data = []
    for _ in tqdm.tqdm(range(random.randint(TOTAL_COUNT, TOTAL_COUNT * 1.2))):
        submission_id = len(submission_data) + 1
        assignment_id = random.choice(assignment_ids)
        student_id = random.choice(student_ids)
        submission_date = random_date(start_date, end_date)
        grade = random.choice(['A', 'B', 'C', 'D', 'F', 'I'])  # Including 'I' for Incomplete

        submission_data.append((submission_id, assignment_id, student_id, submission_date, grade))

    # Insert data into the Submission table
    insert_query = "INSERT INTO Submission (submission_id, assignment_id, student_id, submission_date, grade) VALUES (?, ?, ?, ?, ?)"
    cursor.executemany(insert_query, submission_data)
    conn.commit()


def _2(TOTAL_COUNT, conn, cursor):
    global fake, fake_u

    # Generate and insert data for remaining tables
    assignments = []
    submissions = []
    schedules = []
    majors = []
    textbooks = []
    course_textbooks = []
    libraries = []
    books = []
    book_loans = []
    research_groups = []
    research_projects = []
    project_members = []
    labs = []
    lab_equipments = []
    equipment_maintenances = []

    # Fetch necessary foreign keys from already populated tables
    cursor.execute("SELECT student_id FROM Student")
    student_ids = cursor.fetchall()
    cursor.execute("SELECT professor_id FROM Professor")
    professor_ids = cursor.fetchall()
    cursor.execute("SELECT course_id FROM Course")
    course_ids = cursor.fetchall()
    cursor.execute("SELECT department_id FROM Department")
    department_ids = cursor.fetchall()

    # Generate data for each table
    for i in tqdm.tqdm(range(TOTAL_COUNT)):
        assignment_id = i + 1
        course_id = random.choice(course_ids)[0]
        title = fake.sentence(nb_words=4)
        due_date = fake.date_between(start_date='today', end_date='+1y')
        assignments.append((assignment_id, course_id, title, due_date))

        schedule_id = i + 1
        professor_id = random.choice(professor_ids)[0]
        room = fake.building_number()
        time_slot = fake.time()
        schedules.append((schedule_id, course_id, professor_id, room, time_slot))

        major_id = i + 1
        name = fake.word().capitalize() + " Major"
        department_id = random.choice(department_ids)[0]
        majors.append((major_id, name, department_id))

        textbook_id = i + 1
        title = fake.sentence(nb_words=3)
        author = fake.name()
        isbn = fake_u.isbn13()
        textbooks.append((textbook_id, title, author, isbn))
        course_textbooks.append((course_id, textbook_id))

        library_id = i + 1
        name = fake.company() + " Library"
        location = fake.address()
        libraries.append((library_id, name, location))

        book_id = i + 1
        title = fake.sentence(nb_words=3)
        author = fake.name()
        isbn = fake_u.isbn13()
        books.append((book_id, title, author, isbn, library_id))

        loan_id = i + 1
        book_id = book_id
        student_id = random.choice(student_ids)[0]
        loan_date = fake.date_between(start_date='-1y', end_date='today')
        return_date = fake.date_between(start_date='today', end_date='+1y')
        book_loans.append((loan_id, book_id, student_id, loan_date, return_date))

        group_id = i + 1
        name = fake.word().capitalize() + " Research"
        focus_area = fake.sentence(nb_words=3)
        research_groups.append((group_id, name, focus_area))

        project_id = i + 1
        title = fake.sentence(nb_words=3)
        group_id = group_id
        start_date = fake.date_between(start_date='-1y', end_date='today')
        end_date = fake.date_between(start_date='today', end_date='+1y')
        research_projects.append((project_id, title, group_id, start_date, end_date))
        project_members.append((project_id, professor_id))

        lab_id = i + 1
        name = fake.company() + " Lab"
        building = fake.building_number()
        labs.append((lab_id, name, building))

        equipment_id = i + 1
        name = fake.word().capitalize() + " Equipment"
        lab_id = lab_id
        lab_equipments.append((equipment_id, name, lab_id))

        maintenance_id = i + 1
        equipment_id = equipment_id
        date = fake.date_between(start_date='-1y', end_date='today')
        details = fake.sentence(nb_words=6)
        equipment_maintenances.append((maintenance_id, equipment_id, date, details))

    # Insert data into tables
    cursor.executemany('INSERT INTO Assignment (assignment_id, course_id, title, due_date) VALUES (?, ?, ?, ?);',
                       assignments)
    cursor.executemany(
        'INSERT INTO Schedule (schedule_id, course_id, professor_id, room, time_slot) VALUES (?, ?, ?, ?, ?);',
        schedules)
    cursor.executemany('INSERT INTO Major (major_id, name, department_id) VALUES (?, ?, ?);', majors)
    cursor.executemany('INSERT INTO Textbook (textbook_id, title, author, isbn) VALUES (?, ?, ?, ?);', textbooks)
    cursor.executemany('INSERT INTO CourseTextbook (course_id, textbook_id) VALUES (?, ?);', course_textbooks)
    cursor.executemany('INSERT INTO Library (library_id, name, location) VALUES (?, ?, ?);', libraries)
    cursor.executemany('INSERT INTO Book (book_id, title, author, isbn, library_id) VALUES (?, ?, ?, ?, ?);', books)
    cursor.executemany(
        'INSERT INTO BookLoan (loan_id, book_id, student_id, loan_date, return_date) VALUES (?, ?, ?, ?, ?);',
        book_loans)
    cursor.executemany('INSERT INTO ResearchGroup (group_id, name, focus_area) VALUES (?, ?, ?);', research_groups)
    cursor.executemany(
        'INSERT INTO ResearchProject (project_id, title, group_id, start_date, end_date) VALUES (?, ?, ?, ?, ?);',
        research_projects)
    cursor.executemany('INSERT INTO ProjectMember (project_id, professor_id) VALUES (?, ?);', project_members)
    cursor.executemany('INSERT INTO Lab (lab_id, name, building) VALUES (?, ?, ?);', labs)
    cursor.executemany('INSERT INTO LabEquipment (equipment_id, name, lab_id) VALUES (?, ?, ?);', lab_equipments)
    cursor.executemany(
        'INSERT INTO EquipmentMaintenance (maintenance_id, equipment_id, date, details) VALUES (?, ?, ?, ?);',
        equipment_maintenances)
    conn.commit()


def _1dpcse(TOTAL_COUNT, conn, cursor):
    global fake, fake_u

    # Generate data for Department
    departments = []
    for i in tqdm.tqdm(range(TOTAL_COUNT), desc="Department"):
        department_id = i + 1
        name = fake.word().capitalize() + " Department"
        building = fake.street_name()
        departments.append((department_id, name, building))
    cursor.executemany('INSERT INTO Department (department_id, name, building) VALUES (?, ?, ?);', departments)

    # Generate data for Professor
    professors = []
    for i in tqdm.tqdm(range(TOTAL_COUNT), desc="Professor"):
        professor_id = i + 1
        name = fake.name()
        email = fake_u.email()
        department = random.choice(departments)[0]
        professors.append((professor_id, name, email, department))
    cursor.executemany('INSERT INTO Professor (professor_id, name, email, department) VALUES (?, ?, ?, ?);', professors)

    # Generate data for Course
    courses = []
    for i in tqdm.tqdm(range(TOTAL_COUNT), desc="Course"):
        course_id = i + 1
        title = fake.sentence(nb_words=3)
        credits = random.randint(1, 4)
        department = random.choice(departments)[0]
        courses.append((course_id, title, credits, department))
    cursor.executemany('INSERT INTO Course (course_id, title, credits, department) VALUES (?, ?, ?, ?);', courses)

    # Generate data for Student
    students = []
    for i in tqdm.tqdm(range(TOTAL_COUNT), desc="Student"):
        student_id = i + 1
        name = fake.name()
        email = fake_u.email()
        date_of_birth = fake.date_of_birth(minimum_age=18, maximum_age=30)
        major = random.choice(departments)[0]  # Assuming major is linked to department
        students.append((student_id, name, email, date_of_birth, major))
    cursor.executemany('INSERT INTO Student (student_id, name, email, date_of_birth, major) VALUES (?, ?, ?, ?, ?);',
                       students)

    # Generate data for Enrollment
    enrollments = []
    for i in tqdm.tqdm(range(TOTAL_COUNT), desc="Enrollment"):
        enrollment_id = i + 1
        student_id = random.choice(students)[0]
        course_id = random.choice(courses)[0]
        semester = random.choice(['Spring', 'Summer', 'Fall', 'Winter']) + ' ' + str(random.randint(2019, 2023))
        grade = random.choice(['A', 'B', 'C', 'D', 'F', None])
        enrollments.append((enrollment_id, student_id, course_id, semester, grade))
    cursor.executemany(
        'INSERT INTO Enrollment (enrollment_id, student_id, course_id, semester, grade) VALUES (?, ?, ?, ?, ?);',
        enrollments)

    # Commit changes and close the connection
    conn.commit()


def gen(TOTAL_COUNT, DB_NAME):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    _1dpcse(TOTAL_COUNT, conn, cursor)
    _2(TOTAL_COUNT, conn, cursor)
    _3(TOTAL_COUNT, conn, cursor)
    conn.close()
