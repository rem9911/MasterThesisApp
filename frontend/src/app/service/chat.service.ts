import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ChatRequest, ChatResponse } from '../models/chat.models';

@Injectable({
  providedIn: 'root',
})
export class ChatService {
  private apiUrl = 'http://localhost:8000/api/message';

  constructor(private http: HttpClient) {}

  sendMessage(
    query: string,
    mode: 'naive' | 'advanced' | 'compare',
  ): Observable<ChatResponse> {
    const payload: ChatRequest = { query, mode };
    return this.http.post<ChatResponse>(this.apiUrl, payload);
  }
}
