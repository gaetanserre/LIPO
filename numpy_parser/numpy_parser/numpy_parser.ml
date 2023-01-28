let _ =
  
  let fun_str = Sys.argv.(1) in
  let lexbuf = Lexing.from_string fun_str in
  Parser.function_expr Lexer.token lexbuf;
