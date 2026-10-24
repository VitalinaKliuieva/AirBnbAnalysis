import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({ providedIn: 'root' })
export class Api {
  constructor(private http: HttpClient) {}
  get<T>(path: string, params: Record<string, string | number | boolean> = {}) {
    return this.http.get<T>('/api/' + path, { params: params as any });
  }
}
