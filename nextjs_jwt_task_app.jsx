// Next.js + TSX scaffold with JWT auth
// ---------------------------------------
// 1. pages/_app.tsx — глобальный wrapper
//----------------------------------------
import '@/styles/globals.css';
import type { AppProps } from 'next/app';
import { AuthProvider } from '@/context/AuthContext';
import Navbar from '@/components/Navbar';

export default function MyApp({ Component, pageProps }: AppProps) {
  return (
    <AuthProvider>
      <Navbar />
      <main className="container mx-auto p-4">
        <Component {...pageProps} />
      </main>
    </AuthProvider>
  );
}

// ---------------------------------------
// 2. context/AuthContext.tsx — хранит токен и роль
//----------------------------------------
import {
  createContext,
  useContext,
  useEffect,
  useState,
  ReactNode,
} from 'react';
import { useRouter } from 'next/router';

interface AuthState {
  token: string | null;
  role: string | null;
  nickname: string | null;
}

interface AuthContextProps extends AuthState {
  login: (token: string) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextProps | null>(null);

const parseToken = (jwt: string) => {
  try {
    const base64 = jwt.split('.')[1];
    const json = JSON.parse(atob(base64));
    return {
      role: json.role as string,
      nickname: json.nickname as string,
    };
  } catch {
    return { role: null, nickname: null };
  }
};

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const router = useRouter();
  const [state, setState] = useState<AuthState>({
    token: null,
    role: null,
    nickname: null,
  });

  const login = (token: string) => {
    localStorage.setItem('token', token);
    const { role, nickname } = parseToken(token);
    setState({ token, role, nickname });
  };

  const logout = () => {
    localStorage.removeItem('token');
    setState({ token: null, role: null, nickname: null });
    router.push('/login');
  };

  useEffect(() => {
    const stored = localStorage.getItem('token');
    if (stored) {
      const { role, nickname } = parseToken(stored);
      setState({ token: stored, role, nickname });
    }
  }, []);

  return (
    <AuthContext.Provider value={{ ...state, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
};

// ---------------------------------------
// 3. utils/withAuth.tsx — HOC для защиты страниц
//----------------------------------------
import { useAuth } from '@/context/AuthContext';
import { useRouter } from 'next/router';
import { useEffect } from 'react';

const withAuth = (Component: any, allowed: string[] = []) => {
  return function Wrapped(props: any) {
    const { token, role } = useAuth();
    const router = useRouter();

    useEffect(() => {
      if (!token) router.replace('/login');
      else if (allowed.length && !allowed.includes(role || ''))
        router.replace('/');
    }, [token, role]);

    if (!token) return null;
    if (allowed.length && !allowed.includes(role || '')) return null;

    return <Component {...props} />;
  };
};

export default withAuth;

// ---------------------------------------
// 4. components/Input.tsx — базовый контрол
//----------------------------------------
import { InputHTMLAttributes } from 'react';

export default function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={`w-full rounded-xl border border-gray-300 p-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 ${props.className}`}
    />
  );
}

// ---------------------------------------
// 5. components/Button.tsx
//----------------------------------------
import { ButtonHTMLAttributes } from 'react';

export default function Button(props: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...props}
      className={`rounded-xl bg-blue-600 px-4 py-2 text-white shadow-md transition hover:bg-blue-700 disabled:opacity-50 ${props.className}`}
    />
  );
}

// ---------------------------------------
// 6. components/Navbar.tsx — меню
//----------------------------------------
import Link from 'next/link';
import { useAuth } from '@/context/AuthContext';
import Button from './Button';

export default function Navbar() {
  const { token, role, nickname, logout } = useAuth();

  return (
    <header className="mb-6 flex items-center justify-between rounded-b-xl bg-white p-4 shadow">
      <h1 className="text-2xl font-bold">TaskManager</h1>
      {token ? (
        <nav className="flex items-center gap-4">
          <Link href="/tasks" className="hover:underline">
            Задачи
          </Link>
          {role === 'admin' && (
            <Link href="/users" className="hover:underline">
              Пользователи
            </Link>
          )}
          <span className="text-gray-600">{nickname}</span>
          <Button onClick={logout}>Выйти</Button>
        </nav>
      ) : (
        <Link href="/login" className="hover:underline">
          Войти
        </Link>
      )}
    </header>
  );
}

// ---------------------------------------
// 7. pages/login.tsx — форма входа
//----------------------------------------
import { FormEvent, useState } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '@/context/AuthContext';
import Input from '@/components/Input';
import Button from '@/components/Button';

export default function LoginPage() {
  const { login } = useAuth();
  const router = useRouter();
  const [form, setForm] = useState({ username: '', password: '' });
  const [error, setError] = useState('');

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    try {
      const res = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      if (!res.ok) throw new Error('Неверные учетные данные');
      const { token } = await res.json();
      login(token);
      router.push('/tasks');
    } catch (err: any) {
      setError(err.message);
    }
  };

  return (
    <div className="mx-auto max-w-md rounded-2xl bg-white p-8 shadow-lg">
      <h2 className="mb-6 text-center text-xl font-semibold">Вход</h2>
      <form onSubmit={submit} className="space-y-4">
        <Input
          placeholder="Логин"
          value={form.username}
          onChange={(e) => setForm({ ...form, username: e.target.value })}
        />
        <Input
          type="password"
          placeholder="Пароль"
          value={form.password}
          onChange={(e) => setForm({ ...form, password: e.target.value })}
        />
        {error && <p className="text-sm text-red-600">{error}</p>}
        <Button type="submit" className="w-full">
          Войти
        </Button>
      </form>
    </div>
  );
}

// ---------------------------------------
// 8. pages/tasks.tsx — список и создание задач
//----------------------------------------
import { FormEvent, useEffect, useState } from 'react';
import withAuth from '@/utils/withAuth';
import { useAuth } from '@/context/AuthContext';
import Input from '@/components/Input';
import Button from '@/components/Button';

interface Task {
  id: number;
  title: string;
  description: string;
}

function TasksPage() {
  const { token } = useAuth();
  const [tasks, setTasks] = useState<Task[]>([]);
  const [form, setForm] = useState({ title: '', description: '' });

  const fetchTasks = async () => {
    const res = await fetch('/api/tasks', {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (res.ok) setTasks(await res.json());
  };

  useEffect(() => {
    fetchTasks();
  }, []);

  const create = async (e: FormEvent) => {
    e.preventDefault();
    const res = await fetch('/api/tasks', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(form),
    });
    if (res.ok) {
      setForm({ title: '', description: '' });
      fetchTasks();
    }
  };

  return (
    <div className="space-y-8">
      <form
        onSubmit={create}
        className="rounded-xl bg-white p-6 shadow-md space-y-4"
      >
        <h2 className="text-lg font-semibold">Новая задача</h2>
        <Input
          placeholder="Заголовок"
          value={form.title}
          onChange={(e) => setForm({ ...form, title: e.target.value })}
        />
        <Input
          placeholder="Описание"
          value={form.description}
          onChange={(e) => setForm({ ...form, description: e.target.value })}
        />
        <Button type="submit">Создать</Button>
      </form>

      <div className="grid gap-4 md:grid-cols-2">
        {tasks.map((t) => (
          <div
            key={t.id}
            className="rounded-xl bg-white p-4 shadow hover:shadow-lg"
          >
            <h3 className="text-lg font-bold">{t.title}</h3>
            <p className="text-gray-600">{t.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default withAuth(TasksPage);

// ---------------------------------------
// 9. pages/users.tsx — только для админов
//----------------------------------------
import { FormEvent, useEffect, useState } from 'react';
import withAuth from '@/utils/withAuth';
import { useAuth } from '@/context/AuthContext';
import Input from '@/components/Input';
import Button from '@/components/Button';

interface User {
  id: number;
  username: string;
  role: string;
}

function UsersPage() {
  const { token } = useAuth();
  const [users, setUsers] = useState<User[]>([]);
  const [form, setForm] = useState({ username: '', password: '', role: 'user' });

  const fetchUsers = async () => {
    const res = await fetch('/api/users', {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (res.ok) setUsers(await res.json());
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  const create = async (e: FormEvent) => {
    e.preventDefault();
    const res = await fetch('/api/users', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(form),
    });
    if (res.ok) {
      setForm({ username: '', password: '', role: 'user' });
      fetchUsers();
    }
  };

  return (
    <div className="space-y-8">
      <form
        onSubmit={create}
        className="rounded-xl bg-white p-6 shadow-md space-y-4"
      >
        <h2 className="text-lg font-semibold">Создать пользователя</h2>
        <Input
          placeholder="Логин"
          value={form.username}
          onChange={(e) => setForm({ ...form, username: e.target.value })}
        />
        <Input
          type="password"
          placeholder="Пароль"
          value={form.password}
          onChange={(e) => setForm({ ...form, password: e.target.value })}
        />
        <select
          className="w-full rounded-xl border border-gray-300 p-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={form.role}
          onChange={(e) => setForm({ ...form, role: e.target.value })}
        >
          <option value="user">user</option>
          <option value="admin">admin</option>
        </select>
        <Button type="submit">Создать</Button>
      </form>

      <div className="rounded-xl bg-white p-6 shadow-md">
        <h2 className="mb-4 text-lg font-semibold">Пользователи</h2>
        <ul className="space-y-2">
          {users.map((u) => (
            <li key={u.id} className="flex justify-between rounded-lg p-2 hover:bg-gray-50">
              <span>{u.username}</span>
              <span className="rounded-full bg-gray-200 px-2 text-sm">{u.role}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default withAuth(UsersPage, ['admin']);

// ---------------------------------------
// 10. pages/index.tsx — редирект
//----------------------------------------
import { useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '@/context/AuthContext';

export default function Home() {
  const { token } = useAuth();
  const router = useRouter();

  useEffect(() => {
    router.replace(token ? '/tasks' : '/login');
  }, [token]);

  return null;
}

// ---------------------------------------
// 11. README (коротко, не код) — как запустить
//----------------------------------------
// - npm install next react react-dom typescript tailwindcss postcss autoprefixer
// - npx tailwindcss init -p && настроить content в tailwind.config.js
// - поместить styles/globals.css с @tailwind директивами
// - добавить скрипты "dev": "next dev" и т.д.
// - создать API-роуты или настроить proxy к вашему бэкенду
// - запустить `npm run dev` и открыть http://localhost:3000
