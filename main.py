import csv
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from abc import ABC, abstractmethod
from typing import List


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
        """Завантаження даних з CSV."""
        df_classes = pd.read_csv(classes_file)
        for _, row in df_classes.iterrows():
            new_class = SchoolClass(int(row['parallel']), row['vertical'])
            self.classes.append(new_class)

        df_students = pd.read_csv(students_file)
        for _, row in df_students.iterrows():
            student = Student(
                row['surname'], row['name'], row['patronymic'],
                int(row['year']), row['gender'], float(row['score'])
            )
            target_class = next((c for c in self.classes
                                 if c.parallel == int(row['parallel'])
                                 and c.vertical == row['vertical']), None)
            if target_class:
                target_class.add_student(student)

    def print_statistics(self, title: str):
        """Виведення статистики на сторінку Streamlit."""
        st.subheader(title)

        total_students = sum(c.get_count() for c in self.classes)

        if total_students == 0:
            st.warning("Школа порожня.")
            return

        boys = sum(sum(1 for s in c.students if s.gender == 'Ч') for c in self.classes)
        girls = total_students - boys
        avg_size = total_students / len(self.classes) if self.classes else 0

        sorted_classes = sorted(self.classes, key=lambda x: x.get_count())
        min_c = sorted_classes[0]
        max_c = sorted_classes[-1]

        # Виводимо красиво списком або метриками
        col1, col2, col3 = st.columns(3)
        col1.metric("Всього учнів", total_students)
        col2.metric("Хлопців", f"{boys / total_students:.1%}")
        col3.metric("Дівчат", f"{girls / total_students:.1%}")

        st.write(f"**Середня наповненість:** {avg_size:.1f}")
        st.write(f"**Максимум:** {max_c.name} ({max_c.get_count()} уч.)")
        st.write(f"**Мінімум:** {min_c.name} ({min_c.get_count()} уч.)")

    def show_plots(self):
        """Побудова графіків через Matplotlib та вивід у Streamlit."""
        st.subheader("Візуалізація даних")

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
            st.error("Немає даних для графіків.")
            return

        # Створюємо фігуру matplotlib
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # a. Розподіл по паралелях
        parallel_counts = df['parallel'].value_counts().sort_index()
        axs[0, 0].bar(parallel_counts.index, parallel_counts.values, color='skyblue')
        axs[0, 0].set_title('Кількість учнів по паралелях')

        # b. Середня кількість по вертикалях
        class_sizes = df.groupby(['parallel', 'vertical']).size().reset_index(name='count')
        vertical_avg = class_sizes.groupby('vertical')['count'].mean()
        axs[0, 1].bar(vertical_avg.index, vertical_avg.values, color='lightgreen')
        axs[0, 1].set_title('Середня кількість учнів по вертикалях')

        # c. Лінійний графік від року народження
        year_counts = df['year'].value_counts().sort_index()
        axs[1, 0].plot(year_counts.index, year_counts.values, marker='o')
        axs[1, 0].set_title('Кількість учнів за роком народження')
        axs[1, 0].grid(True)

        # d. Scatter: середня оцінка vs клас
        axs[1, 1].scatter(df['parallel'], df['score'], alpha=0.5, c='orange')
        axs[1, 1].set_title('Розподіл оцінок по паралелях')
        axs[1, 1].set_xlabel('Паралель')
        axs[1, 1].set_ylabel('Оцінка')

        plt.tight_layout()

        # Головна зміна: передаємо фігуру в Streamlit
        st.pyplot(fig)

    def perform_graduation(self):
        """Переведення на рік вперед."""
        new_classes = []
        for c in self.classes:
            if c.parallel == 11:
                continue
            c.parallel += 1
            new_classes.append(c)

        self.classes = new_classes
        st.success("✅ Переведення класів на наступний рік виконано успішно!")


# ==========================================
# ЧАСТИНА 2: СПІВРОБІТНИКИ
# ==========================================

class Employee(ABC):
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
        return (self.base_salary * self.ped_exp / 50) + (self.man_exp * 500)


class Teacher(Employee):
    def __init__(self, name: str, base_salary: float, ped_exp: int):
        super().__init__(name, base_salary)
        self.ped_exp = ped_exp

    def calculate_salary(self) -> float:
        return self.base_salary * self.ped_exp / 30


class SecurityGuard(Employee):
    def __init__(self, name: str, base_salary: float, work_exp: int):
        super().__init__(name, base_salary)
        self.work_exp = work_exp

    def calculate_salary(self) -> float:
        return self.base_salary + (self.work_exp * 250)


# ==========================================
# ГОЛОВНИЙ БЛОК (STREAMLIT LOGIC)
# ==========================================

# Налаштування сторінки
st.set_page_config(page_title="Шкільна система", layout="wide")
st.title("🎓 Система керування школою")

# 1. Завантаження (Сценарій 1)
school = School()
# Streamlit перезапускає скрипт при кожній дії, тому вантажимо дані щоразу
school.load_data("classes.csv", "students.csv")

# 2. Статистика ДО переведення
school.print_statistics("Статистика (Поточний рік)")

# 3. Графіки
school.show_plots()

# 4. Переведення
st.markdown("---")
st.header("Переведення на наступний рік")
if st.button("Виконати переведення класів"):
    school.perform_graduation()
    # 5. Статистика ПІСЛЯ переведення
    school.print_statistics("Статистика (Наступний рік)")
else:
    st.info("Натисніть кнопку вище, щоб перевести учнів у наступні класи.")

# --- СЦЕНАРІЙ 2: ЗАРПЛАТИ ---
st.markdown("---")
st.header("💰 Розрахунок зарплат (Сценарій 2)")

employees = [
    Director("Петренко П.П.", 15000, ped_exp=20, man_exp=5),
    Teacher("Іваненко І.І.", 12000, ped_exp=10),
    Teacher("Сидорова С.С.", 12000, ped_exp=25),
    SecurityGuard("Коваленко К.К.", 11000, work_exp=5)
]

salary_data = []
for emp in employees:
    sal = emp.calculate_salary()
    salary_data.append({"ПІБ": emp.name, "Посада": type(emp).__name__, "Зарплата (грн)": round(sal, 2)})

# Вивід таблиці в Streamlit
df_salary = pd.DataFrame(salary_data)
st.dataframe(df_salary, use_container_width=True)

# Збереження
if st.button("Зберегти зарплати у CSV"):
    df_salary.to_csv("salaries.csv", index=False)
    st.success("Файл 'salaries.csv' успішно збережено!")