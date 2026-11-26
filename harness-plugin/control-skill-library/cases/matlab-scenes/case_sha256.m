function text=case_sha256(file)
% 按原始字节计算摘要，避免编码及同名文件差异影响结果追溯。
fid=fopen(file,'rb');assert(fid>=0);c=onCleanup(@()fclose(fid)); %#ok<NASGU>
bytes=fread(fid,inf,'*uint8');md=java.security.MessageDigest.getInstance('SHA-256');
text=lower(reshape(dec2hex(typecast(md.digest(bytes),'uint8'),2).',1,[]));
end
