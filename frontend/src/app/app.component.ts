import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatService } from './service/chat.service';
import { ChatMessage } from './models/chat.models';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  messages: ChatMessage[] = [
    { 
      content: "Hello! I am your legal AI assistant. How can I help you with Consumer Law?", 
      sender: 'bot', 
      timestamp: new Date() 
    }
  ];
  userInput: string = '';
  isLoading: boolean = false;
  currentMode: 'naive' | 'advanced' | 'compare' = 'advanced';

  constructor(private chatService: ChatService) {}

  sendMessage() {
    if (!this.userInput.trim()) return;

    const query = this.userInput;
    this.messages.push({ content: query, sender: 'user', timestamp: new Date() });
    this.userInput = '';
    this.isLoading = true;

    this.chatService.sendMessage(query, this.currentMode).subscribe({
      next: (resp) => {
        // Comparison mode
        if (resp.comparison) {
          this.messages.push({
            sender: 'bot',
            timestamp: new Date(),
            comparisonData: resp.comparison
          });
        } 
        // Standard mode
        else {
          this.messages.push({
            content: resp.answer,
            sender: 'bot',
            sources: resp.sources,
            timestamp: new Date()
          });
        }
        this.isLoading = false;
      },
      error: (err) => {
        console.error(err);
        this.messages.push({ content: "Server error.", sender: 'bot', timestamp: new Date() });
        this.isLoading = false;
      }
    });
  }
}