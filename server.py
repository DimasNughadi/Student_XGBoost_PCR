from flask import Flask, flash, render_template, request, redirect, url_for, session, Markup, jsonify, make_response, Response
from flask_mysqldb import MySQL
import MySQLdb.cursors
import mysql.connector
import secrets
import json
import hashlib
import pandas as pd
import pickle
from sklearn import preprocessing
import pdfkit
import os
import hashlib
import subprocess

app = Flask(__name__)

app.secret_key = secrets.token_hex(16)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'kelulusan'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['UPLOAD_IMG'] = 'static/foto/'
app.config['ALLOWED_EXTENSIONS'] = {'xls', 'xlsx', 'csv'}
app.config['SESSION_TYPE'] = 'filesystem'

mysql = MySQL(app)

# Load the saved model using pickle.load()
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('processed_df2.pkl', 'rb') as f:
    processed_df2 = pickle.load(f)

with open("time_output.pkl", "rb") as f:
    time_output = pickle.load(f)

@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify({'error': 'Bad request'}), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/', methods=['POST', 'GET'])
def auth():
    title = 'Login'
    return render_template("login.html", title=title)

@app.route('/regist', methods=['POST', 'GET'])
def regist():
    title = 'Registrasi'

    cursor = mysql.connection.cursor()
    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']
        password = hashlib.md5(request.form['password'].encode())
        no_hp = request.form['no_hp']
        alamat = request.form['alamat']
        foto = request.files['foto']
        filename = foto.filename
        filepath = os.path.join(app.config['UPLOAD_IMG'], filename)
        foto.save(filepath)
        role = request.form['role']
        password2 = password.hexdigest()

        cursor.execute(
            '''INSERT INTO user (email, username, password, no_hp, alamat, foto, role) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)''',(email, username, password2, no_hp, alamat, filename, role))
        mysql.connection.commit()
        mysql.connection.close()
        msg = Markup('<div class="alert alert-success" role="alert">Selamat <b>' +
                     username + ' </b>Anda Berhasil Mendaftar!</div>')
        flash(msg)
        return redirect(url_for('login'))

    return render_template("register.html", title=title)

@app.route('/login', methods=['POST', 'GET'])
def login():
    title = 'Login'

    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        pw_enc = hashlib.md5(password.encode()).hexdigest()
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            'SELECT * FROM user WHERE email = % s AND password = % s', (email, pw_enc,))
        user = cursor.fetchone()
        if user:
            session['loggedin'] = True
            session['id'] = user['id_user']
            session['email'] = user['email']
            session['username'] = user['username']
            session['no_hp'] = user['no_hp']
            session['alamat'] = user['alamat']
            session['foto'] = user['foto']
            session['role'] = user['role']
            username = session["username"]

            msg = Markup('<div class="alert alert-success" role="alert">Selamat <b>' 
                         + username + ' </b>Anda Berhasil Login!</div>')
            flash(msg)
            return redirect(url_for('dashboard'))
        else:
            msg = Markup(
                '<div class="alert alert-danger" role="alert">Email / Password Salah!</div>')
            flash(msg)
    return render_template('login.html', title=title)

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('email', None)

    msg = Markup(
        '<div class="alert alert-success" role="alert">Berhasil Logout!</div>')
    flash(msg)
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['POST', 'GET'])
def dashboard():
    title = 'Home'

    if "email" in session:
        cursor = mysql.connection.cursor()
        total_alumni = ("Select count(NIM) from alumni")
        cursor.execute(total_alumni)
        data = cursor.fetchall()

        total_mahasiswa = ("Select count(NIM) from mahasiswa")
        cursor.execute(total_mahasiswa)
        data2 = cursor.fetchall()

        total_tepat = ("Select count(NIM) from alumni where STATUS = 'Tepat Waktu'")
        cursor.execute(total_tepat)
        data_tepat = cursor.fetchall()

        total_terlambat = ("Select count(NIM) from alumni where STATUS = 'Tidak Tepat Waktu'")
        cursor.execute(total_terlambat)
        data3 = cursor.fetchall()

        total_tepat_mhs = (
            "Select count(NIM) from mahasiswa where STATUS = 'Tepat Waktu'")
        cursor.execute(total_tepat_mhs)
        data_tepat_mhs = cursor.fetchall()

        total_terlambat_mhs = (
            "Select count(NIM) from mahasiswa where STATUS = 'Tidak Tepat Waktu'")
        cursor.execute(total_terlambat_mhs)
        data_terlambat_mhs = cursor.fetchall()

        data_terlambat = (
            "Select * from mahasiswa where STATUS = 'Tidak Tepat Waktu'")
        cursor.execute(data_terlambat)
        data4 = cursor.fetchall()

        data_alumni = processed_df2

        X = data_alumni.drop("STATUS", axis=1)
        y = data_alumni["STATUS"]

        accuracyy = model.score(X, y)
        accuracy = ("%.2f%%" % (accuracyy * 100.0))

        total_perbandingan = ('''SELECT LEFT(NIM, 4),
        COUNT(CASE WHEN STATUS = 'Tepat Waktu' THEN NIM END) AS tepat_waktu_count,
        COUNT(CASE WHEN STATUS = 'Tidak Tepat Waktu' THEN NIM END) AS tidak_tepat_waktu_count,
        FORMAT((COUNT(CASE WHEN STATUS = 'Tepat Waktu' THEN NIM END) / 
        CAST(COUNT(NIM) AS FLOAT)) * 100, 2) AS tepat_waktu_percentage
        FROM alumni
        GROUP BY LEFT(NIM, 4)''')

        cursor.execute(total_perbandingan)
        data_perbandingan = cursor.fetchall()

        labels = []
        data_comp = []
        status = []
        for row in data_perbandingan:
            labels.append(row[0])
            data_comp.append(row[3])
            status.append(row[2])

        chart_data = {
            'labels': labels,
            'datasets': [{
                'label': 'Persentase',
                'data': data_comp,
                'backgroundColor':"#28a745",
                'borderColor': '#012749'
            }]
        }

        dataa = [data, data2]
        data_status = [data_tepat, data3]
        data_prediksi = [data_tepat_mhs, data_terlambat_mhs]
        return render_template("index.html", title=title, data=data, data2=data2, data3=data3, data4=data4, 
                               dataa=json.dumps(dataa), data_status=json.dumps(data_status), data_prediksi=json.dumps(data_prediksi), 
                               accuracy=accuracy, data_terlambat_mhs=data_terlambat_mhs, chart_data=json.dumps(chart_data),)
    
    return render_template("login.html")

@app.route('/alumni', methods=['GET'])
def alumni():
    title = 'Data Alumni'
    cursor = mysql.connection.cursor()
    tahun = ("select distinct LEFT(NIM, 4) from alumni")
    cursor.execute(tahun)
    tahun = cursor.fetchall()

    selected_year = request.args.get('tahun')
    if selected_year:
        cursor.execute(
            'SELECT * FROM alumni WHERE LEFT(NIM, 4)=%s', (selected_year,))
    else:
        cursor.execute('SELECT * FROM alumni')
    data = cursor.fetchall()

    total_tepat = (
        "Select count(NIM) from alumni where STATUS = 'Tepat Waktu'")
    cursor.execute(total_tepat)
    data_tepat = cursor.fetchall()

    total_terlambat = (
        "Select count(NIM) from alumni where STATUS = 'Tidak Tepat Waktu'")
    cursor.execute(total_terlambat)
    data_terlambat = cursor.fetchall()

    time = ("", time_output["fit_time"])
    time2 = (time[0] + "{:.2f}".format(time[1]))

    return render_template("dataAlumni.html", title=title, data=data, data_tepat=data_tepat, data_terlambat=data_terlambat, waktu=time2, tahun=tahun)

@app.route('/mahasiswa', methods=['GET'])
def mahasiswa():
    title = 'Data Mahasiswa'

    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM mahasiswa")
    data = cursor.fetchall()

    total_tepat = (
        "Select count(NIM) from mahasiswa where STATUS = 'Tepat Waktu'")
    cursor.execute(total_tepat)
    data_tepat = cursor.fetchall()

    total_terlambat = (
        "Select count(NIM) from mahasiswa where STATUS = 'Tidak Tepat Waktu'")
    cursor.execute(total_terlambat)
    data_terlambat = cursor.fetchall()

    time = ("", time_output["fit_time"])
    time2 = (time[0] + "{:.2f}".format(time[1]))

    kelas = ("select distinct KELAS from mahasiswa ORDER BY KELAS ASC")
    cursor.execute(kelas)
    kelas = cursor.fetchall()

    return render_template("dataMahasiswa.html", title=title, data=data, data_tepat=data_tepat, data_terlambat=data_terlambat, waktu=time2, kelas=kelas)

@app.route('/akun', methods=['GET'])
def akun():
    title = 'Akun'
    user_id = session.get('id')

    cursor = mysql.connection.cursor()

    query = "SELECT * FROM user WHERE id_user = %s"
    cursor.execute(query, (user_id,))
    data = cursor.fetchall()

    return render_template("akun.html", title=title, data=data)

@app.route('/upload_alumni', methods=['POST'])
def upload_alumni():

    if request.method == 'POST':
        if 'file' not in request.files:
            msg = Markup('<div class="alert alert-danger" role="alert">File tidak Ditemukan!</div>')
            flash(msg)
            return redirect(url_for('alumni'))
        file = request.files['file']

        if file.filename == '':
            msg = Markup(
                '<div class="alert alert-danger" role="alert">File tidak Ditemukan!</div>')
            flash(msg)
            return redirect(url_for('alumni'))
        if file and allowed_file(file.filename):

            df = pd.read_excel(file)
            df = df.dropna()

            cursor = mysql.connection.cursor()

            # loop through each row in the dataframe and insert it into the MySQL database
            for index, row in df.iterrows():
                sql = "INSERT INTO alumni (NIM, NAMA, KELAS, CUTI, KP, IPS1, IPS2, IPS3, IPS4, IPS5, IPS6, CO, KOMPEN, TAK, STATUS) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                val = (row['NIM'], row['NAMA'], row['KELAS'], row['CUTI'], row['KP'], row['IPS1'], row['IPS2'],
                       row['IPS3'], row['IPS4'], row['IPS5'], row['IPS6'], row['CO'], row['KOMPEN'], row['TAK'], row['STATUS'])
                cursor.execute(sql, val)
            mysql.connection.commit()
            mysql.connection.close()
            # subprocess.run(['python', '/XGBoost.py'], stdout=subprocess.PIPE)
            # output = result.stdout.decode()
            # return output
            msg = Markup('<div class="alert alert-success" role="alert">Berhasil Menambah Data Alumni</div>')
            flash(msg)

            return redirect(url_for('alumni'))
        msg = Markup('<div class="alert alert-danger" role="alert">File Harus Berupa Excel atau csv</div>')
        flash(msg)

        return redirect(url_for('alumni'))

@app.route('/upload_mahasiswa', methods=['POST'])
def upload_mahasiswa():
    
    if request.method == 'POST':
        if 'file' not in request.files:
            msg = Markup('<div class="alert alert-danger" role="alert">File tidak Ditemukan!</div>')
            flash(msg)

            return redirect(url_for('mahasiswa'))
        
        file = request.files['file']
        if file.filename == '':
            msg = Markup('<div class="alert alert-danger" role="alert">File tidak Ditemukan!</div>')
            flash(msg)

            return redirect(url_for('mahasiswa'))
        
        if file and allowed_file(file.filename):

            df = pd.read_excel(file)
            df = df.dropna()

            cursor = mysql.connection.cursor()

            # loop through each row in the dataframe and insert it into the MySQL database
            for index, row in df.iterrows():
                sql = '''INSERT INTO mahasiswa (NIM, NAMA, KELAS, CUTI, KP, IPS1, IPS2, IPS3, IPS4, IPS5, IPS6, CO, KOMPEN, TAK) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'''
                val = (row['NIM'], row['NAMA'], row['KELAS'], row['CUTI'], row['KP'], row['IPS1'], row['IPS2'], row['IPS3'], row['IPS4'],
                       row['IPS5'], row['IPS6'], row['CO'], row['KOMPEN'], row['TAK'])
                cursor.execute(sql, val)

            cursor.execute("SELECT * FROM mahasiswa")
            mydata = cursor.fetchall()

            data_prediksi = pd.DataFrame(mydata, columns=[
                                         'NIM', 'NAMA', 'KELAS', 'CUTI', 'KP', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'IPS6', 'CO', 'KOMPEN', 'TAK', 'STATUS'])
            data_proses1 = preprocesss_kelulusan_df(data_prediksi)
            data_proses2 = preprocess_kelulusan_df2(data_proses1)

            columns2 = ['CUTI', 'KP', 'KATIPS1', 'KATIPS2', 'KATIPS3', 'KATIPS4', 'KATIPS5', 'KATIPS6', 'CO', 'KATKOMPEN', 'KATTAK']
            data_proses2 = data_proses2.reindex(columns=columns2)

            prediksi = model.predict(data_proses2)


            # Define the class labels
            class_labels = {1: 'Tidak Tepat Waktu', 0: 'Tepat Waktu'}
            class_labels_predicted = [class_labels[p]for p in prediksi]

            data_prediksi['STATUS'] = class_labels_predicted

            for index, row in data_prediksi.iterrows():
                sql = "UPDATE mahasiswa SET STATUS = %s WHERE NIM = %s"
                val = (row['STATUS'], row['NIM'])
                cursor.execute(sql, val)

            mysql.connection.commit()
            mysql.connection.close()

            msg = Markup('<div class="alert alert-success" role="alert">Berhasil Menambah Data Mahasiswa</div>')
            flash(msg)

            return redirect(url_for('mahasiswa'))

    msg = Markup('<div class="alert alert-danger" role="alert">File Harus Berupa Excel atau csv</div>')
    flash(msg)
    
    return redirect(url_for('mahasiswa'))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower(
           ) in app.config['ALLOWED_EXTENSIONS']

@app.route('/prediksi', methods=['GET', 'POST'])
def prediksi():
    title = 'Prediksi'
    return render_template("prediksi.html", title=title)

@app.route('/predict', methods=['POST'])
def predict():
    prediksi = ''

    if request.method == 'POST':

        mydata = [{
            'NIM': '00000000',
            'NAMA': '',
            'CUTI': request.form['cuti'],
            'KP': request.form['kp'],
            'IPS1': float(request.form['ips1']),
            'IPS2': float(request.form['ips2']),
            'IPS3': float(request.form['ips3']),
            'IPS4': float(request.form['ips4']),
            'IPS5': float(request.form['ips5']),
            'IPS6': float(request.form['ips6']),
            'CO': float(request.form['co']),
            'KOMPEN': float(request.form['kompen']),
            'TAK': float(request.form['tak']),
        }]

        data_prediksi = pd.DataFrame(mydata, columns=['NIM', 'NAMA', 'CUTI', 'KP', 'IPS1', 'IPS2', 'IPS3',
                                                      'IPS4', 'IPS5', 'IPS6', 'CO', 'KOMPEN', 'TAK'])

        data_proses1 = preprocesss_kelulusan_df(data_prediksi)
        data_proses2 = preprocess_kelulusan_df3(data_proses1)

        columns2 = ['CUTI', 'KP', 'KATIPS1', 'KATIPS2', 'KATIPS3', 'KATIPS4',
                    'KATIPS5', 'KATIPS6', 'CO', 'KATKOMPEN', 'KATTAK']
        data_proses2 = data_proses2.reindex(columns=columns2)

        prediksi2 = model.predict(data_proses2)

        hasil = ""
        if prediksi2 == '0':
            hasil = "Tepat Waktu"
        elif prediksi2 == '1':
            hasil = "Tidak Tepat Waktu"

        # msg = (f'<div class="alert alert-success" role="alert">Prediksi Berhasil {prediksi2} </div>')
        msg = (f'Hasil Prediksi {hasil}!')
        flash(msg)
        return redirect(url_for('prediksi'))

def preprocesss_kelulusan_df(df_raw):
    df_raw.loc[df_raw['IPS1'] <= 2.76, 'KATIPS1'] = '<2.76'
    df_raw.loc[(df_raw['IPS1'] > 2.76) & (
        df_raw['IPS1'] <= 3), 'KATIPS1'] = '2.76 - 3.00'
    df_raw.loc[(df_raw['IPS1'] > 3) & (
        df_raw['IPS1'] <= 3.5), 'KATIPS1'] = '3.00 - 3.50'
    df_raw.loc[df_raw['IPS1'] > 3.5, 'KATIPS1'] = '3.50 - 4.00'

    df_raw.loc[df_raw['IPS2'] <= 2.76, 'KATIPS2'] = '<2.76'
    df_raw.loc[(df_raw['IPS2'] > 2.76) & (
        df_raw['IPS2'] <= 3), 'KATIPS2'] = '2.76 - 3.00'
    df_raw.loc[(df_raw['IPS2'] > 3) & (
        df_raw['IPS2'] <= 3.5), 'KATIPS2'] = '3.00 - 3.50'
    df_raw.loc[df_raw['IPS2'] > 3.5, 'KATIPS2'] = '3.50 - 4.00'

    df_raw.loc[df_raw['IPS3'] <= 2.76, 'KATIPS3'] = '<2.76'
    df_raw.loc[(df_raw['IPS3'] > 2.76) & (
        df_raw['IPS3'] <= 3), 'KATIPS3'] = '2.76 - 3.00'
    df_raw.loc[(df_raw['IPS3'] > 3) & (
        df_raw['IPS3'] <= 3.5), 'KATIPS3'] = '3.00 - 3.50'
    df_raw.loc[df_raw['IPS3'] > 3.5, 'KATIPS3'] = '3.50 - 4.00'

    df_raw.loc[df_raw['IPS4'] <= 2.76, 'KATIPS4'] = '<2.76'
    df_raw.loc[(df_raw['IPS4'] > 2.76) & (
        df_raw['IPS4'] <= 3), 'KATIPS4'] = '2.76 - 3.00'
    df_raw.loc[(df_raw['IPS4'] > 3) & (
        df_raw['IPS4'] <= 3.5), 'KATIPS4'] = '3.00 - 3.50'
    df_raw.loc[df_raw['IPS4'] > 3.5, 'KATIPS4'] = '3.50 - 4.00'

    df_raw.loc[df_raw['IPS5'] <= 2.76, 'KATIPS5'] = '<2.76'
    df_raw.loc[(df_raw['IPS5'] > 2.76) & (
        df_raw['IPS5'] <= 3), 'KATIPS5'] = '2.76 - 3.00'
    df_raw.loc[(df_raw['IPS5'] > 3) & (
        df_raw['IPS5'] <= 3.5), 'KATIPS5'] = '3.00 - 3.50'
    df_raw.loc[df_raw['IPS5'] > 3.5, 'KATIPS5'] = '3.50 - 4.00'

    df_raw.loc[df_raw['IPS6'] <= 2.76, 'KATIPS6'] = '<2.76'
    df_raw.loc[(df_raw['IPS6'] > 2.76) & (
        df_raw['IPS6'] <= 3), 'KATIPS6'] = '2.76 - 3.00'
    df_raw.loc[(df_raw['IPS6'] > 3) & (
        df_raw['IPS6'] <= 3.5), 'KATIPS6'] = '3.00 - 3.50'
    df_raw.loc[df_raw['IPS6'] > 3.5, 'KATIPS6'] = '3.50 - 4.00'

    df_raw.loc[df_raw['KOMPEN'] <= 25, 'KATKOMPEN'] = '<25'
    df_raw.loc[(df_raw['KOMPEN'] > 25) & (
        df_raw['KOMPEN'] <= 50), 'KATKOMPEN'] = '25 - 50'
    df_raw.loc[(df_raw['KOMPEN'] > 50) & (
        df_raw['KOMPEN'] <= 75), 'KATKOMPEN'] = '50 - 75'
    df_raw.loc[df_raw['KOMPEN'] > 75, 'KATKOMPEN'] = '>75'

    df_raw.loc[df_raw['TAK'] <= 25, 'KATTAK'] = '<25'
    df_raw.loc[(df_raw['TAK'] > 25) & (
        df_raw['TAK'] <= 50), 'KATTAK'] = '25 - 50'
    df_raw.loc[(df_raw['TAK'] > 50) & (
        df_raw['TAK'] <= 75), 'KATTAK'] = '50 - 75'
    df_raw.loc[df_raw['TAK'] > 75, 'KATTAK'] = '>75'

    return df_raw

def preprocess_kelulusan_df2(df_raw):
    processed_df = df_raw.copy()
    le = preprocessing.LabelEncoder()
    processed_df.CUTI = le.fit_transform(processed_df.CUTI)
    processed_df.KP = le.fit_transform(processed_df.KP)
    processed_df.KATIPS1 = le.fit_transform(processed_df.KATIPS1)
    processed_df.KATIPS2 = le.fit_transform(processed_df.KATIPS2)
    processed_df.KATIPS3 = le.fit_transform(processed_df.KATIPS3)
    processed_df.KATIPS4 = le.fit_transform(processed_df.KATIPS4)
    processed_df.KATIPS5 = le.fit_transform(processed_df.KATIPS5)
    processed_df.KATIPS6 = le.fit_transform(processed_df.KATIPS6)
    processed_df.KATKOMPEN = le.fit_transform(processed_df.KATKOMPEN)
    processed_df.KATTAK = le.fit_transform(processed_df.KATTAK)
    processed_df = processed_df.drop(["NIM", "NAMA", "STATUS"], axis=1)

    return processed_df

def preprocess_kelulusan_df3(df_raw):
    processed_df = df_raw.copy()
    le = preprocessing.LabelEncoder()
    processed_df.CUTI = le.fit_transform(processed_df.CUTI)
    processed_df.KP = le.fit_transform(processed_df.KP)
    processed_df.KATIPS1 = le.fit_transform(processed_df.KATIPS1)
    processed_df.KATIPS2 = le.fit_transform(processed_df.KATIPS2)
    processed_df.KATIPS3 = le.fit_transform(processed_df.KATIPS3)
    processed_df.KATIPS4 = le.fit_transform(processed_df.KATIPS4)
    processed_df.KATIPS5 = le.fit_transform(processed_df.KATIPS5)
    processed_df.KATIPS6 = le.fit_transform(processed_df.KATIPS6)
    processed_df.KATKOMPEN = le.fit_transform(processed_df.KATKOMPEN)
    processed_df.KATTAK = le.fit_transform(processed_df.KATTAK)
    processed_df = processed_df.drop(["NIM", "NAMA"], axis=1)

    return processed_df

@app.route('/export_alumni', methods=['GET', 'POST'])
def export_alumni():
    cursor = mysql.connection.cursor()
    selected_year = request.args.get('tahun')
    if selected_year:
        cursor.execute(
            'SELECT * FROM alumni WHERE LEFT(NIM, 4)=%s', (selected_year,))
    else:
        cursor.execute('SELECT * FROM alumni')
    data = cursor.fetchall()

    html = render_template("tabel_alumni.html", data=data)
    options = {
        'page-size': 'A4',
        'orientation': 'Landscape',
        'enable-local-file-access': ''
    }

    filename = f'Data_Lulusan_TI_{selected_year}.pdf'

    pdf = pdfkit.from_string(html, False, options=options)
    response = make_response(pdf)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"

    return response

@app.route('/export_mhs', methods=['GET', 'POST'])
def export_mhs():
    cursor = mysql.connection.cursor()
    selected_class = request.args.get('kelas')
    if selected_class:
        cursor.execute(
            'SELECT * FROM mahasiswa WHERE KELAS=%s', (selected_class,))
    else:
        cursor.execute('SELECT * FROM mahasiswa')
    data = cursor.fetchall()

    html = render_template("tabel_mahasiswa.html", data=data)
    options = {
        'page-size': 'A4',
        'orientation': 'Landscape',
        'enable-local-file-access': ''
    }

    filename = f'Data_Prediksi_Kelulusan_{selected_class}.pdf'

    pdf = pdfkit.from_string(html, False, options=options)
    response = make_response(pdf)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"

    return response

@app.route('/export_alumni_excel', methods=['GET', 'POST'])
def export_alumni_excel():
    cursor = mysql.connection.cursor()
    selected_year = request.args.get('tahun')
    if selected_year:
        cursor.execute('SELECT * FROM alumni WHERE LEFT(NIM, 4)=%s', (selected_year,))
    else:
        cursor.execute('SELECT * FROM alumni')
    data = cursor.fetchall()
    
    # cursor.execute("SELECT * FROM alumni")
    # data = cursor.fetchall()
    data = pd.DataFrame(data, columns=[i[0] for i in cursor.description])

    # Create a new Excel file using pandas
    writer = pd.ExcelWriter('export.xlsx', engine='xlsxwriter')
    data.to_excel(writer, index=False)
    writer.save()
    
    filename = f'Data_Lulusan_TI_{selected_year}.xlsx'

    # Send the Excel file as a response to the client
    with open('export.xlsx', 'rb') as file:
        data = file.read()
    response = Response(data, headers={
        'Content-Disposition': f'attachment;filename={filename}',
        'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    })
    return response

@app.route('/export_mhs_excel', methods=['GET', 'POST'])
def export_mhs_excel():
    cursor = mysql.connection.cursor()
    selected_class = request.args.get('kelas')
    if selected_class:
        cursor.execute(
            'SELECT * FROM mahasiswa WHERE KELAS=%s', (selected_class,))
    else:
        cursor.execute('SELECT * FROM mahasiswa')
    data = cursor.fetchall()

    # cursor.execute("SELECT * FROM alumni")
    # data = cursor.fetchall()
    data = pd.DataFrame(data, columns=[i[0] for i in cursor.description])

    # Create a new Excel file using pandas
    writer = pd.ExcelWriter('export.xlsx', engine='xlsxwriter')
    data.to_excel(writer, index=False)
    writer.save()

    filename = f'Data_Prediksi_Kelulusan_{selected_class}.xlsx'

    # Send the Excel file as a response to the client
    with open('export.xlsx', 'rb') as file:
        data = file.read()
    response = Response(data, headers={
        'Content-Disposition': f'attachment;filename={filename}',
        'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    })
    return response


@app.route('/template_alumni', methods=['GET', 'POST'])
def template_alumni():
    # Define the headings
    headings = ['NIM', 'NAMA', 'KELAS', 'CUTI', 'KP', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'IPS6', 'CO', 'KOMPEN', 'TAK', 'STATUS']

    # Create an empty DataFrame with the headings
    data = pd.DataFrame(columns=headings)

    # Create a new Excel file using pandas
    writer = pd.ExcelWriter('export.xlsx', engine='xlsxwriter')
    data.to_excel(writer, index=False)
    writer.save()

    # Set the filename for the Excel file
    filename = 'Template_Alumni.xlsx'

    # Send the Excel file as a response to the client
    with open('export.xlsx', 'rb') as file:
        data = file.read()
    response = Response(data, headers={
        'Content-Disposition': f'attachment;filename={filename}',
        'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    })
    return response


@app.route('/template_mhs', methods=['GET', 'POST'])
def template_mhs():
    # Define the headings
    headings = ['NIM', 'NAMA', 'KELAS', 'CUTI', 'KP', 'IPS1', 'IPS2',
                'IPS3', 'IPS4', 'IPS5', 'IPS6', 'CO', 'KOMPEN', 'TAK']

    # Create an empty DataFrame with the headings
    data = pd.DataFrame(columns=headings)

    # Create a new Excel file using pandas
    writer = pd.ExcelWriter('export.xlsx', engine='xlsxwriter')
    data.to_excel(writer, index=False)
    writer.save()

    # Set the filename for the Excel file
    filename = 'Template_Mahasiswa.xlsx'

    # Send the Excel file as a response to the client
    with open('export.xlsx', 'rb') as file:
        data = file.read()
    response = Response(data, headers={
        'Content-Disposition': f'attachment;filename={filename}',
        'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    })
    return response

@app.route("/hapus")
def hapus():
    cursor = mysql.connection.cursor()
    cursor.execute("DELETE FROM mahasiswa")
    cursor.execute("DELETE FROM alumni")

    mysql.connection.commit()
    mysql.connection.close()

    return redirect(url_for('dashboard'))

@app.route('/riwayat')
def riwayat():
    title = 'Riwayat Prediksi'

    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM alumni")
    data = cursor.fetchall()

    total_tepat = (
        "Select count(NIM) from alumni where STATUS = 'Tepat Waktu'")
    cursor.execute(total_tepat)
    data_tepat = cursor.fetchall()

    total_terlambat = (
        "Select count(NIM) from alumni where STATUS = 'Tidak Tepat Waktu'")
    cursor.execute(total_terlambat)
    data_terlambat = cursor.fetchall()

    total_avg = (
        '''SELECT 
        CONCAT(ROUND((COUNT(CASE WHEN STATUS = 'Tepat Waktu' THEN 1 END) / COUNT(*) * 100), 2), '%') AS tepat_waktu_percentage 
        FROM 
        alumni
        '''
    )
    cursor.execute(total_avg)
    data_avg = cursor.fetchall()

    total_perbandingan = ('''SELECT LEFT(NIM, 4),
        COUNT(CASE WHEN STATUS = 'Tepat Waktu' THEN NIM END) AS tepat_waktu_count,
        COUNT(CASE WHEN STATUS = 'Tidak Tepat Waktu' THEN NIM END) AS tidak_tepat_waktu_count,
        FORMAT((COUNT(CASE WHEN STATUS = 'Tepat Waktu' THEN NIM END) / CAST(COUNT(NIM) AS FLOAT)) * 100, 2) AS tepat_waktu_percentage
        FROM alumni
        GROUP BY LEFT(NIM, 4)'''
        )
    cursor.execute(total_perbandingan)
    data_perbandingan = cursor.fetchall()

    time = ("", time_output["fit_time"])
    time2 = (time[0] + "{:.2f}".format(time[1]))

    return render_template("riwayat.html", title=title, data=data, data_tepat=data_tepat, data_terlambat=data_terlambat, waktu=time2, data_perbandingan=data_perbandingan, data_avg=data_avg)

if (__name__ == '__main__'):
    app.run(host="192.168.100.7", port=5000, debug=True)
