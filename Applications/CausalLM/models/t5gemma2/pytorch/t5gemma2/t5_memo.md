2. Component Inheritance Tree                                                                   
     Gemma3 Components (reused)                                        
     ├── Gemma3PreTrainedModel ← T5Gemma2PreTrainedModel                
     ├── Gemma3RMSNorm ← T5Gemma2RMSNorm                                          
     ├── Gemma3MLP ← T5Gemma2MLP  (added dropout)                                                                     
     ├── Gemma3RotaryEmbedding ← T5Gemma2RotaryEmbedding                 
     ├── Gemma3Attention ← T5Gemma2SelfAttention (encoder)                
     ├── Gemma3Attention ← T5Gemma2MergedAttention (decoder,fused)        
     ├── Gemma3MultiModalProjector ← T5Gemma2MultiModalProjector          
     ├── Gemma3TextScaledWordEmbedding ← T5Gemma2TextScaledWordEmbedding  
     └── Gemma3Config ← T5Gemma2EncoderConfig, T5Gemma2TextConfig, T5Gemma2DecoderConfig

     T5Gemma Components (reused)                                     
     ├── T5GemmaEncoderLayer ← T5Gemma2EncoderLayer                     
     ├── T5GemmaEncoderLayer ← T5Gemma2DecoderLayer                       
     ├── T5GemmaLMHead ← T5Gemma2LMHead                                   
     └── T5GemmaClassificationHead ← T5Gemma2ClassificationHead (unused in main)


     5:1 sliding/full attn

     tie weight sharing씀


생성시 생기는 일 : 
Summary - What Gets Updated                                                                                              
                                                                                                                              
     | Component | Updated Each Step? | Why |                                                                                 
     |-----------|-------------------|-----|                                                                                  
     | Encoder | ❌ NO | Context doesn't change - "describe this image" stays the same |                                      
     | Encoder hidden states | ❌ NO | Computed once, cached as cross_attention_cache |                                       
     | Decoder input | ✅ YES | Grows: <bos> → <bos>A → <bos>A bee → ... |
     | Decoder cross-attention cache | ❌ NO | Encoder K,V never changes |                                                    
                                                                                                                              
     ---                                                                                                                      
                                                                                                                              
     Performance Impact                                                                                                       
                                                                                                                              
     Without caching:                                                                                                         
       Step 1: Encode (cost: E) + Decode (cost: D₁)                                                                           
       Step 2: Encode (cost: E) + Decode (cost: D₂)                                                                           
       Step 3: Encode (cost: E) + Decode (cost: D₃)                                                                           
       ...
        Total = N×E + ΣDᵢ                                                                                                      
                                                                                                                              
     With caching:                                                                                                            
       Step 0: Encode (cost: E) → Cache!                                                                                      
       Step 1: Decode (cost: D₁, use cache)                                                                                   
       Step 2: Decode (cost: D₂, use cache)                                                                                   
       Step 3: Decode (cost: D₃, use cache)                                                                                   
       ...                                                                                                                    
       Total = E + ΣDᵢ                                                                                                        
                                                                                                                              
     Speedup: ~2-3× faster for long sequences because encoder is recomputed ~0 times instead of N times!