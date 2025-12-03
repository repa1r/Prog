import csv
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Dict


# ==========================================
# ЧАСТИНА 1: СУТНОСТІ ШКОЛИ
# ==========================================

class Student:
    """Клас, що описує учня."""

    def __init__(self, surname: str, name: str, patronymic: str,
                 year: int, gender: str, score: float):
        self.surname = surname
        self.name = name
        self.patronymic = patronymic
        self.year = year
        self.gender = gender
        self.score = score

    def __repr__(self):
        return f"{self.surname} {self.name}"


class SchoolClass:
    """Клас, що описує навчальний клас (наприклад, 10-Б)."""

    def __init__(self, parallel: int, vertical: str):
        self.parallel = parallel
        self.vertical = vertical
        self.students: List[Student] = []

    def add_student(self, student: Student):
        self.students.append(student)

    @property
    def name(self) -> str:
        return f"{self.parallel}-{self.vertical}"

    def get_count(self) -> int:
        return len(self.students)


class School:
    """Головний клас для керування школою."""

    def __init__(self):
        self.classes: List[SchoolClass] = []

    def load_data(self, classes_file: str, students_file: str):
        """Завантаження даних з CSV та розподіл учнів по об'єктах класів."""
        # 1. Створюємо класи
        df_classes = pd.read_csv(classes_file)
        for _, row in df_classes.iterrows():
            new_class = SchoolClass(int(row['parallel']), row['vertical'])
            self.classes.append(new_class)

        # 2. Створюємо учнів і шукаємо їм клас
        df_students = pd.read_csv(students_file)
        for _, row in df_students.iterrows():
            student = Student(
                row['surname'], row['name'], row['patronymic'],
                int(row['year']), row['gender'], float(row['score'])
            )

            # Шукаємо потрібний клас у списку (за паралеллю та вертикаллю)
            target_class = next((c for c in self.classes
                                 if c.parallel == int(row['parallel'])
                                 and c.vertical == row['vertical']), None)

            if target_class:
                target_class.add_student(student)

    def print_statistics(self):
        """Виведення текстової статистики згідно з пунктом 2."""
        total_students = sum(c.get_count() for c in self.classes)

        if total_students == 0:
            print("Школа порожня.")
            return

        # Рахуємо хлопців/дівчат
        boys = sum(sum(1 for s in c.students if s.gender == 'Ч') for c in self.classes)
        girls = total_students - boys

        # Середня наповненість
        avg_size = total_students / len(self.classes) if self.classes else 0

        # Мін/Макс класи
        # Сортуємо класи за кількістю учнів
        sorted_classes = sorted(self.classes, key=lambda x: x.get_count())
        min_c = sorted_classes[0]
        max_c = sorted_classes[-1]

        print(f"--- Статистика ---")
        print(f"a. Всього учнів: {total_students}")
        print(f"b. Хлопців: {boys / total_students:.1%}, Дівчат: {girls / total_students:.1%}")
        print(f"c. Середнє в класі: {avg_size:.1f}")
        print(f"d. Максимум: {max_c.name} ({max_c.get_count()} уч.)")
        print(f"e. Мінімум: {min_c.name} ({min_c.get_count()} уч.)")

    def show_plots(self):
        """Побудова графіків (пункт 3). Переганяємо об'єкти в DataFrame для зручності."""
        data = []
        for c in self.classes:
            for s in c.students:
                data.append({
                    'parallel': c.parallel,
                    'vertical': c.vertical,
                    'year': s.year,
                    'score': s.score
                })

        df = pd.DataFrame(data)
        if df.empty:
            return

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # a. Розподіл по паралелях
        parallel_counts = df['parallel'].value_counts().sort_index()
        axs[0, 0].bar(parallel_counts.index, parallel_counts.values, color='skyblue')
        axs[0, 0].set_title('Кількість учнів по паралелях')

        # b. Середня кількість по вертикалях (А, Б, В...)
        # Групуємо спочатку по класах, потім по вертикалі
        class_sizes = df.groupby(['parallel', 'vertical']).size().reset_index(name='count')
        vertical_avg = class_sizes.groupby('vertical')['count'].mean()
        axs[0, 1].bar(vertical_avg.index, vertical_avg.values, color='lightgreen')
        axs[0, 1].set_title('Середня кількість учнів по вертикалях')

        # c. Лінійний графік від року народження
        year_counts = df['year'].value_counts().sort_index()
        axs[1, 0].plot(year_counts.index, year_counts.values, marker='o')
        axs[1, 0].set_title('Кількість учнів за роком народження')
        axs[1, 0].grid(True)

        # d. Scatter: середня оцінка vs клас (паралель)
        # Використовуємо середній бал конкретного учня, як на схемі, чи середній по класу?
        # В завданні "середньої оцінки учнів від класу". Зробимо scatter всіх учнів.
        axs[1, 1].scatter(df['parallel'], df['score'], alpha=0.5, c='orange')
        axs[1, 1].set_title('Розподіл оцінок по паралелях')
        axs[1, 1].set_xlabel('Паралель')
        axs[1, 1].set_ylabel('Оцінка')

        plt.tight_layout()
        plt.show()

    def perform_graduation(self):
        """Переведення на рік вперед (пункт 4)."""
        new_classes = []
        for c in self.classes:
            # 11-ті класи випускаються (зникають)
            if c.parallel == 11:
                continue

            # Інші переходять далі
            c.parallel += 1
            new_classes.append(c)

        self.classes = new_classes
        print("\n=== ВІДБУЛОСЯ ПЕРЕВЕДЕННЯ НА НАСТУПНИЙ РІК ===")
        # 1-х класів немає, бо ми їх не набирали (згідно умови)


# ==========================================
# ЧАСТИНА 2: СПІВРОБІТНИКИ
# ==========================================

class Employee(ABC):
    """Базовий абстрактний клас працівника."""

    def __init__(self, name: str, base_salary: float):
        self.name = name
        self.base_salary = base_salary

    @abstractmethod
    def calculate_salary(self) -> float:
        pass


class Director(Employee):
    def __init__(self, name: str, base_salary: float, ped_exp: int, man_exp: int):
        super().__init__(name, base_salary)
        self.ped_exp = ped_exp
        self.man_exp = man_exp

    def calculate_salary(self) -> float:
        # Формула з ТЗ: base * ped / 50 + man * 500
        return (self.base_salary * self.ped_exp / 50) + (self.man_exp * 500)


class Teacher(Employee):
    def __init__(self, name: str, base_salary: float, ped_exp: int):
        super().__init__(name, base_salary)
        self.ped_exp = ped_exp

    def calculate_salary(self) -> float:
        # Формула з ТЗ: base * ped / 30
        return self.base_salary * self.ped_exp / 30


class SecurityGuard(Employee):
    def __init__(self, name: str, base_salary: float, work_exp: int):
        super().__init__(name, base_salary)
        self.work_exp = work_exp

    def calculate_salary(self) -> float:
        # Формула з ТЗ: base + exp * 250
        return self.base_salary + (self.work_exp * 250)


# ==========================================
# ЗАПУСК (SCENARIO 1 & 2)
# ==========================================

if __name__ == "__main__":
    # --- СЦЕНАРІЙ 1 ---
    print("Завантаження даних...")
    school = School()
    school.load_data("classes.csv", "students.csv")

    # 2. Статистика до переведення
    school.print_statistics()

    # 3. Графіки
    print("Відображення графіків...")
    school.show_plots()

    # 4. Переведення
    school.perform_graduation()

    # 5. Статистика після переведення
    school.print_statistics()

    # --- СЦЕНАРІЙ 2 ---
    print("\n--- Розрахунок зарплат ---")
    employees = [
        Director("Петренко П.П.", 15000, ped_exp=20, man_exp=5),
        Teacher("Іваненко І.І.", 12000, ped_exp=10),
        Teacher("Сидорова С.С.", 12000, ped_exp=25),
        SecurityGuard("Коваленко К.К.", 11000, work_exp=5)
    ]

    salary_data = []
    for emp in employees:
        sal = emp.calculate_salary()
        salary_data.append([emp.name, type(emp).__name__, round(sal, 2)])

    # Збереження у CSV
    with open("salaries.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Role", "Salary"])
        writer.writerows(salary_data)

    print("Зарплати розраховано та збережено у 'salaries.csv'.")