export interface ChatRequest {
  query: string;
  mode: string;
}

export interface Source {
    article_number: string;
    content: string;
    score: number;
    metadata?: any;
}

export interface ChatResult {
    answer: string;
    sources: Source[];
    processing_time: number;
}

export interface ChatResponse {
    answer?: string;
    sources?: Source[];
    processing_time?: number;
    comparison?: {
        naive: ChatResult;
        advanced: ChatResult;
    };
}

export interface ChatMessage {
    content?: string;
    sender: 'user' | 'bot';
    timestamp: Date;
    sources?: Source[];
    isStreaming?: boolean;
    comparisonData?: {
        naive: ChatResult;
        advanced: ChatResult;
    };
}